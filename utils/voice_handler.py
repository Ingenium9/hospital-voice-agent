import os
import base64
import threading
import speech_recognition as sr
from elevenlabs.client import ElevenLabs


class VoiceHandler:
    def __init__(self):
        try:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            self.is_listening = False
            self.callback = None
            self.client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

            print("Calibrating microphone...")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            print("Microphone calibrated")
        except Exception as e:
            print("Microphone initialization failed:", e)
            raise

    def start_continuous_listening(self, callback):
        if self.is_listening:
            print("Already listening")
            return

        self.is_listening = True
        self.callback = callback

        def listen_loop():
            while self.is_listening:
                try:
                    with self.microphone as source:
                        audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=8)
                    text = self.recognizer.recognize_google(audio)
                    if text and len(text.strip()) > 2 and self.callback:
                        print("User:", text)
                        self.callback(text)
                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    continue
                except Exception as e:
                    print("Listening error:", e)
                    continue

        self.listening_thread = threading.Thread(target=listen_loop, daemon=True)
        self.listening_thread.start()
        print("Continuous listening started")

    def stop_listening(self):
        self.is_listening = False
        print("Listening stopped")

    def text_to_speech(self, text, voice="Rachel"):
        if not text or not text.strip():
            return None
        try:
            print("Generating speech...")
            audio = self.client.generate(text=text, voice=voice, model="eleven_monolingual_v1")
            audio_data = b""
            for chunk in audio:
                audio_data += chunk
            audio_base64 = base64.b64encode(audio_data).decode("utf-8")
            return audio_base64
        except Exception as e:
            print("Text-to-speech error:", e)
            return None
