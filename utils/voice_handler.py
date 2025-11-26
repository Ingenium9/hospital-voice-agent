import speech_recognition as sr
from elevenlabs.client import ElevenLabs
import base64
import threading
import time
import os

class VoiceHandler:
    def __init__(self):
        try:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            self.is_listening = False
            self.callback = None
            self.client = ElevenLabs(api_key=os.getenv('ELEVENLABS_API_KEY'))
            
            print("🎤 Calibrating microphone...")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            print("✅ Microphone calibrated")
            
        except Exception as e:
            print(f"❌ Microphone initialization failed: {e}")
            raise
    
    def start_continuous_listening(self, callback):
        """Start continuous listening"""
        if self.is_listening:
            print("⚠️ Already listening")
            return
            
        self.is_listening = True
        self.callback = callback
        
        def listen_loop():
            while self.is_listening:
                try:
                    with self.microphone as source:
                        audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=8)
                    
                    text = self.recognizer.recognize_google(audio)
                    
                    if text and len(text.strip()) > 2:
                        print(f"🗣️ User: {text}")
                        if self.callback:
                            self.callback(text)
                            
                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    continue
                except Exception as e:
                    print(f"❌ Listening error: {e}")
                    continue
        
        self.listening_thread = threading.Thread(target=listen_loop)
        self.listening_thread.daemon = True
        self.listening_thread.start()
        print("✅ Continuous listening started")
    
    def stop_listening(self):
        """Stop continuous listening"""
        self.is_listening = False
        print("🛑 Listening stopped")
    
    def text_to_speech(self, text, voice="Rachel"):
        """Convert text to speech"""
        try:
            if not text or len(text.strip()) == 0:
                return None
                
            print(f"🔊 Generating speech: {text[:50]}...")
            
            # Generate audio using the latest API
            audio = self.client.generate(
                text=text,
                voice=voice,
                model="eleven_monolingual_v1"
            )
            
            # Convert to base64
            audio_data = b""
            for chunk in audio:
                audio_data += chunk
                
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            print("✅ Speech generated")
            return audio_base64
            
        except Exception as e:
            print(f"❌ Text-to-speech error: {e}")
            return None