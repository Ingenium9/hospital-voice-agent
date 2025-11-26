class VoiceAgent {
  constructor() {
    this.socket = io();
    this.micButton = document.getElementById('micButton');
    this.status = document.getElementById('status');
    this.conversationLog = document.getElementById('conversationLog');
    this.audioPlayer = document.getElementById('audioPlayer');
    this.visualizer = document.getElementById('visualizer');
    this.systemStatus = document.getElementById('systemStatus');

    this.isRecording = false;
    this.mediaRecorder = null;
    this.mediaStream = null;
    this.mediaTimesliceMs = 5000;
    this.autoStopTimer = null;

    this.recognition = null;
    this.hasWebSpeech = false;

    this.synth = window.speechSynthesis || null;

    this.setupSocketListeners();
    this.initializeEventListeners();
    this.updateStatus('ready');
    this.initWebSpeechRecognition();
  }

  // Socket listeners
  setupSocketListeners() {
    this.socket.on('connect', () => console.log('Socket connected'));
    this.socket.on('system_status', (data) => this.systemStatusUpdate(data));
    this.socket.on('assistant_response', (data) => {
      const text = data.text || '';
      this.addMessage('assistant', text);
      if (data.audio) this.playBase64Audio(data.audio);
      else this.speakClientSide(text);
    });
    this.socket.on('user_input', (data) => {
      if (data && data.text) this.addMessage('user', data.text);
    });
    this.socket.on('error', (d) => {
      console.error('Server error:', d);
      this.showError(d.message || 'Server error');
    });
  }

  // UI events
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
      if (this.mediaStream) this.mediaStream.getTracks().forEach(t => t.stop());
      if (this.recognition && this.recognition.stop) {
        try { this.recognition.onend = null; this.recognition.stop(); } catch (e) {}
      }
    });
  }

  systemStatusUpdate(data) {
    const initialized = !!(data && data.initialized);
    if (initialized) {
      this.systemStatus.innerHTML = '<span class="status-badge ready">System Ready</span>';
      this.micButton.disabled = false;
    } else {
      this.systemStatus.innerHTML = '<span class="status-badge error">System Issues</span>';
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

  showError(msg) {
    console.error('UI Error:', msg);
    this.status.textContent = `Error: ${msg}`;
    setTimeout(() => this.updateStatus('ready'), 3500);
  }

  // Web Speech API initialization
  initWebSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition || null;
    if (!SpeechRecognition) {
      this.hasWebSpeech = false;
      return;
    }
    try {
      this.recognition = new SpeechRecognition();
      this.recognition.continuous = true;
      this.recognition.interimResults = false;
      this.recognition.lang = 'en-IN';
      this.recognition.onstart = () => console.log('SpeechRecognition started');
      this.recognition.onend = () => {
        if (this.isRecording) {
          try { this.recognition.start(); } catch (e) { console.log('recognition restart failed', e); }
        }
      };
      this.recognition.onerror = () => {};
      this.recognition.onresult = (evt) => {
        let finalText = '';
        for (let i = evt.resultIndex; i < evt.results.length; ++i) {
          const res = evt.results[i];
          if (res.isFinal) finalText += res[0].transcript;
        }
        finalText = finalText.trim();
        if (finalText) this.socket.emit('text_input', { text: finalText });
      };
      this.hasWebSpeech = true;
    } catch (e) {
      this.hasWebSpeech = false;
    }
  }

  // Start/stop listening
  async startRecording() {
    if (this.isRecording) return;
    this.isRecording = true;
    this.micButton.classList.add('listening');
    this.visualizer.classList.add('active');
    this.updateStatus('listening');

    if (this.hasWebSpeech && this.recognition) {
      try {
        try { this.recognition.onend = () => { if (this.isRecording) { try { this.recognition.start(); } catch (e) {} } }; } catch (e) {}
        this.recognition.start();
        this.socket.emit('listening_status', { status: 'active' });
      } catch (e) {
        this.hasWebSpeech = false;
        await this._startMediaRecorderFallback();
      }
    } else {
      await this._startMediaRecorderFallback();
    }

    this.socket.emit('toggle_conversation', { action: 'start' });
  }

  async _startMediaRecorderFallback() {
    try {
      if (!this.mediaStream) {
        this.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      }
      if (!this.mediaRecorder) {
        try {
          this.mediaRecorder = new MediaRecorder(this.mediaStream);
        } catch (err) {
          this.mediaRecorder = new MediaRecorder(this.mediaStream, { mimeType: 'audio/webm' });
        }
        this.mediaRecorder.ondataavailable = (e) => {
          if (e.data && e.data.size > 0) this._uploadAudioChunk(e.data);
        };
      }

      this.mediaRecorder.start(this.mediaTimesliceMs);
      if (this.autoStopTimer) clearTimeout(this.autoStopTimer);
      this.autoStopTimer = setTimeout(() => {
        if (this.isRecording) this.stopRecording();
      }, 5 * 60 * 1000);
    } catch (e) {
      this.showError('Microphone access failed: ' + (e.message || e));
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
      try { this.recognition.onend = null; this.recognition.stop(); } catch (e) {}
    }
    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
      try { this.mediaRecorder.stop(); } catch (e) {}
    }
    if (this.autoStopTimer) {
      clearTimeout(this.autoStopTimer);
      this.autoStopTimer = null;
    }
    this.updateStatus('ready');
    this.socket.emit('toggle_conversation', { action: 'stop' });
  }

  // Upload recorded chunk for server transcription
  async _uploadAudioChunk(blob) {
    if (!blob || blob.size === 0) return;
    this.addMessage('user', '(voice message recorded)');
    try {
      const fd = new FormData();
      fd.append('file', blob, 'chunk.webm');
      this.updateStatus('processing');
      const resp = await fetch('/transcribe', { method: 'POST', body: fd });
      this.updateStatus('listening');
      if (!resp.ok) {
        const txt = await resp.text();
        console.error('Transcribe failed:', resp.status, txt);
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

  // Play base64 audio or fallback to speechSynthesis
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
