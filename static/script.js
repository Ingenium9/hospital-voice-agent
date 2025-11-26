// static/script.js — continuous-conversation ready version
// Replace the old static/script.js with this file.

class VoiceAgent {
  constructor() {
    this.socket = io();
    this.micButton = document.getElementById('micButton');
    this.status = document.getElementById('status');
    this.conversationLog = document.getElementById('conversationLog');
    this.audioPlayer = document.getElementById('audioPlayer');
    this.visualizer = document.getElementById('visualizer');
    this.systemStatus = document.getElementById('systemStatus');

    // Recording flags and objects
    this.isRecording = false;
    this.mediaRecorder = null;
    this.chunks = [];
    this.mediaStream = null;
    this.mediaTimesliceMs = 5000; // for fallback MediaRecorder chunk size
    this.autoStopTimer = null;

    // Web Speech API recognition
    this.recognition = null;
    this.hasWebSpeech = false;

    // speech synthesis fallback
    this.synth = window.speechSynthesis || null;
    this.currentUtterance = null;

    this.setupSocketListeners();
    this.initializeEventListeners();
    this.updateStatus('ready');
    this.initWebSpeechRecognition();
    this.log('VoiceAgent ready');
  }

  // ---------- socket listeners ----------
  setupSocketListeners() {
    this.socket.on('connect', () => this.log('Connected to server'));

    this.socket.on('system_status', (data) => {
      this.systemStatusUpdate(data);
    });

    this.socket.on('assistant_response', (data) => {
      const text = data.text || '';
      this.addMessage('assistant', text);
      if (data.audio) {
        this.playBase64Audio(data.audio);
      } else {
        this.speakClientSide(text);
      }
    });

    this.socket.on('user_input', (data) => {
      if (data && data.text) this.addMessage('user', data.text);
    });

    this.socket.on('error', (d) => {
      console.error('Server error:', d);
      this.showError(d.message || 'Server error');
    });
  }

  // ---------- init / UI ----------
  initializeEventListeners() {
    this.micButton.addEventListener('click', () => {
      if (this.isRecording) this.stopRecording();
      else this.startRecording();
    });

    this.audioPlayer.addEventListener('play', () => {
      this.visualizer.classList.add('speaking');
      this.updateStatus('speaking');
    });
    this.audioPlayer.addEventListener('ended', () => {
      this.visualizer.classList.remove('speaking');
      this.updateStatus('ready');
    });

    window.addEventListener('beforeunload', () => {
      if (this.synth && this.synth.speaking) this.synth.cancel();
      if (this.mediaStream) {
        this.mediaStream.getTracks().forEach(t => t.stop());
      }
      if (this.recognition && this.recognition.stop) {
        try { this.recognition.onend = null; this.recognition.stop(); } catch(e){}
      }
    });
  }

  systemStatusUpdate(data) {
    const initialized = !!(data && data.initialized);
    if (initialized) {
      this.systemStatus.innerHTML = '<span class="status-badge ready">✅ System Ready</span>';
      this.micButton.disabled = false;
    } else {
      this.systemStatus.innerHTML = '<span class="status-badge error">❌ System Issues</span>';
      this.micButton.disabled = true;
    }
  }

  updateStatus(s) {
    const messages = {
      ready: 'Click microphone to start conversation',
      listening: 'Listening... click to stop',
      processing: 'Processing... sending audio to server',
      speaking: 'Playing reply...'
    };
    this.status.textContent = messages[s] || s;
  }

  addMessage(sender, text) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = (sender === 'user' ? 'You: ' : '') + text;
    messageDiv.appendChild(contentDiv);
    this.conversationLog.appendChild(messageDiv);
    this.conversationLog.scrollTop = this.conversationLog.scrollHeight;
  }

  // ---------- logging / UI helpers ----------
  log(...args) {
    console.log(...args);
  }

  showError(msg) {
    console.error('UI Error:', msg);
    this.status.textContent = `Error: ${msg}`;
    setTimeout(() => this.updateStatus('ready'), 3500);
  }

  // ---------- Web Speech API (preferred for continuous) ----------
  initWebSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition || null;
    if (!SpeechRecognition) {
      this.log('Web Speech API not available; will use MediaRecorder fallback');
      this.hasWebSpeech = false;
      return;
    }
    try {
      this.recognition = new SpeechRecognition();
      this.recognition.continuous = true;
      this.recognition.interimResults = false;
      this.recognition.lang = 'en-IN';
      this.recognition.onstart = () => this.log('SpeechRecognition started');
      this.recognition.onend = () => {
        this.log('SpeechRecognition ended');
        // if we are still in continuous mode and user hasn't stopped, restart
        if (this.isRecording) {
          try { this.recognition.start(); } catch (e) { this.log('recognition restart failed', e); }
        }
      };
      this.recognition.onerror = (e) => {
        this.log('SpeechRecognition error', e);
        // Some errors are recoverable; others we stop continuous mode and fallback
        // If fatal, stop recognition and fallback
      };
      this.recognition.onresult = (evt) => {
        // send all final results
        let finalText = '';
        for (let i = evt.resultIndex; i < evt.results.length; ++i) {
          const res = evt.results[i];
          if (res.isFinal) finalText += res[0].transcript;
        }
        finalText = finalText.trim();
        if (finalText) {
          //this.addMessage('user', finalText);
          this.socket.emit('text_input', { text: finalText });
        }
      };
      this.hasWebSpeech = true;
      this.log('Web Speech API initialized (continuous mode available)');
    } catch (e) {
      this.log('SpeechRecognition init failed:', e);
      this.hasWebSpeech = false;
    }
  }

  // ---------- start / stop continuous listening ----------
  async startRecording() {
    if (this.isRecording) return;
    this.isRecording = true;
    this.micButton.classList.add('listening');
    this.visualizer.classList.add('active');
    this.updateStatus('listening');

    // If Web Speech API available -> use it (best, with near real-time results)
    if (this.hasWebSpeech && this.recognition) {
      try {
        // Avoid start() InvalidStateError when already started
        try { this.recognition.onend = () => { if (this.isRecording) { try { this.recognition.start(); } catch(e){} } }; } catch(e){}
        this.recognition.start();
        this.log('Conversation started via SpeechRecognition');
        // tell server we started (optional)
        this.socket.emit('listening_status', { status: 'active' });
        // emit greeting once (server may already send greeting on start via socket)
      } catch (e) {
        this.log('SpeechRecognition start error, falling back to MediaRecorder:', e);
        this.hasWebSpeech = false;
        await this._startMediaRecorderFallback();
      }
    } else {
      // Start MediaRecorder fallback (record short timeslice chunks repeatedly)
      await this._startMediaRecorderFallback();
    }

    // Send toggle to server (so server can optionally send initial greeting)
    this.socket.emit('toggle_conversation', { action: 'start' });
  }

  async _startMediaRecorderFallback() {
    try {
      // get mic stream
      if (!this.mediaStream) {
        this.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      }
      // create MediaRecorder if necessary
      if (!this.mediaRecorder) {
        try {
          this.mediaRecorder = new MediaRecorder(this.mediaStream);
        } catch (err) {
          // fallback to webm if browser needs type
          this.mediaRecorder = new MediaRecorder(this.mediaStream, { mimeType: 'audio/webm' });
        }

        this.mediaRecorder.ondataavailable = (e) => {
          if (e.data && e.data.size > 0) {
            // send chunk for transcription (non-blocking)
            this._uploadAudioChunk(e.data);
          }
        };
        this.mediaRecorder.onstop = () => {
          this.log('MediaRecorder stopped');
        };
      }

      // start with timeslice; this triggers ondataavailable periodically
      // Use timeslice to produce chunks while keeping continuous recording.
      this.mediaRecorder.start(this.mediaTimesliceMs); // e.g., every 5s chunk generated
      this.log('MediaRecorder started (timeslice)', this.mediaTimesliceMs);
      // Safety: auto-stop after long time (5 minutes)
      if (this.autoStopTimer) clearTimeout(this.autoStopTimer);
      this.autoStopTimer = setTimeout(() => {
        if (this.isRecording) {
          this.log('Auto-stop safety triggered');
          this.stopRecording();
        }
      }, 5 * 60 * 1000);
    } catch (e) {
      this.showError('Microphone access failed: ' + e.message);
      console.error(e);
      this.isRecording = false;
      this.micButton.classList.remove('listening');
      this.visualizer.classList.remove('active');
      this.updateStatus('ready');
    }
  }

  stopRecording() {
    if (!this.isRecording) return;
    this.isRecording = false;
    this.micButton.classList.remove('listening');
    this.visualizer.classList.remove('active');
    if (this.recognition) {
      try {
        // Remove onend handler to avoid auto-restart
        this.recognition.onend = null;
        this.recognition.stop();
      } catch (e) { /* ignore */ }
    }
    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
      try { this.mediaRecorder.stop(); } catch (e) { /* ignore */ }
    }
    if (this.autoStopTimer) {
      clearTimeout(this.autoStopTimer);
      this.autoStopTimer = null;
    }
    this.updateStatus('ready');
    // inform server
    this.socket.emit('toggle_conversation', { action: 'stop' });
    this.log('Conversation stopped by user');
  }

  // ---------- upload chunk (MediaRecorder fallback) ----------
  async _uploadAudioChunk(blob) {
    if (!blob || blob.size === 0) return;
    // show interim user bubble
    this.addMessage('user', '(voice message recorded)');
    // POST to /transcribe (same server endpoint)
    try {
      const fd = new FormData();
      fd.append('file', blob, 'chunk.webm');
      this.updateStatus('processing');
      const resp = await fetch('/transcribe', { method: 'POST', body: fd });
      this.updateStatus('listening');
      if (!resp.ok) {
        const txt = await resp.text();
        console.error('Transcribe failed: ', resp.status, txt);
        return;
      }
      const j = await resp.json();
      const text = j.text || '';
      if (text) {
        this.addMessage('user', text);
        this.socket.emit('text_input', { text });
      }
    } catch (e) {
      console.error('Upload/Transcribe failed:', e);
    } finally {
      this.updateStatus(this.isRecording ? 'listening' : 'ready');
    }
  }

  // ---------- audio playback & fallback speak ----------
  playBase64Audio(b64) {
    if (!b64) return;
    try {
      const byteCharacters = atob(b64);
      const byteNumbers = new Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) byteNumbers[i] = byteCharacters.charCodeAt(i);
      const byteArray = new Uint8Array(byteNumbers);
      const blob = new Blob([byteArray], { type: 'audio/mpeg' });
      const url = URL.createObjectURL(blob);
      this.audioPlayer.src = url;
      this.audioPlayer.play().catch(err => {
        console.warn('Audio play blocked:', err);
        // fallback to local TTS
        this.speakClientSide(this.getLatestAssistantText());
      });
    } catch (e) {
      console.error('playBase64Audio error', e);
      this.speakClientSide(this.getLatestAssistantText());
    }
  }

  getLatestAssistantText() {
    const nodes = this.conversationLog.querySelectorAll('.assistant-message .message-content');
    if (!nodes || nodes.length === 0) return '';
    return nodes[nodes.length - 1].textContent || '';
  }

  speakClientSide(text) {
    if (!text || !this.synth) return;
    try {
      // cancel any existing speech to prefer latest
      if (this.synth.speaking) this.synth.cancel();
      const u = new SpeechSynthesisUtterance(text);
      u.lang = 'en-IN';
      u.rate = 1.0;
      u.pitch = 1.0;
      const voices = this.synth.getVoices();
      if (voices && voices.length) {
        const preferred = voices.find(v => /en-?in|india|english/i.test(v.name)) || voices[0];
        if (preferred) u.voice = preferred;
      }
      u.onstart = () => { this.visualizer.classList.add('speaking'); this.updateStatus('speaking'); };
      u.onend = () => { this.visualizer.classList.remove('speaking'); this.updateStatus('ready'); };
      u.onerror = (e) => { console.warn('speak error', e); this.updateStatus('ready'); };
      this.synth.speak(u);
    } catch (e) {
      console.error('speakClientSide error', e);
    }
  }
}

document.addEventListener('DOMContentLoaded', () => {
  console.log('Hospital Voice Agent initializing...');
  new VoiceAgent();
});
