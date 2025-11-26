# twilio_server.py
import os
import time
import logging
from threading import Event

from flask import Flask, request, Response, url_for
from twilio.twiml.voice_response import VoiceResponse, Gather
import socketio  # python-socketio client

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("twilio_server")

# Config
MAIN_APP_SOCKETIO_URL = os.getenv("MAIN_APP_SOCKETIO_URL", "http://localhost:5000")
ASSISTANT_RESPONSE_TIMEOUT = float(os.getenv("ASSISTANT_RESPONSE_TIMEOUT", "8.0"))
TWILIO_LISTEN_PROMPT = os.getenv("TWILIO_LISTEN_PROMPT",
                                 "Hello. Please say your hospital query after the beep. I will try to help you.")

app = Flask(__name__)

class AssistantBridge:
    """
    Creates a transient Socket.IO client to forward captured text to the main app
    and wait for one assistant_response event. This keeps the Twilio webhook flow simple.
    """
    def __init__(self, main_url):
        self.sio = socketio.Client(logger=False, engineio_logger=False, reconnection=False)
        self._event = Event()
        self._response_text = None
        self.connected = False

        @self.sio.event
        def connect():
            log.info("Socket.IO connected to main app")
            self.connected = True

        @self.sio.event
        def disconnect():
            log.info("Socket.IO disconnected")
            self.connected = False

        @self.sio.on("assistant_response")
        def on_assistant_response(data):
            try:
                text = (data or {}).get("text", "")
                log.info("assistant_response received: %s", text)
                self._response_text = text
                self._event.set()
            except Exception as e:
                log.exception("assistant_response handler error: %s", e)

        self.main_url = main_url

    def connect(self, timeout=3.0):
        try:
            self.sio.connect(self.main_url, wait=True, wait_timeout=timeout, transports=["websocket", "polling"])
            return True
        except Exception as e:
            log.exception("Socket.IO connect failed: %s", e)
            return False

    def disconnect(self):
        try:
            if self.sio.connected:
                self.sio.disconnect()
        except Exception:
            pass

    def send_and_wait_response(self, user_text, timeout=ASSISTANT_RESPONSE_TIMEOUT):
        """
        Emit `text_input` to the main app and wait for an `assistant_response` event.
        Returns the assistant text or None on timeout/error.
        """
        if not self.connect():
            log.error("Could not connect to main app Socket.IO at %s", self.main_url)
            return None

        # clear any previous state
        self._event.clear()
        self._response_text = None

        try:
            log.info("Emitting text_input: %s", user_text)
            self.sio.emit("text_input", {"text": user_text})
            waited = self._event.wait(timeout)
            if waited and self._response_text:
                return self._response_text
            else:
                log.info("No assistant_response arrived within %.1fs", timeout)
                return None
        except Exception as e:
            log.exception("Error while waiting for assistant response: %s", e)
            return None
        finally:
            try:
                self.disconnect()
            except Exception:
                pass


@app.route("/twilio/voice", methods=["GET", "POST"])
def twilio_voice_entry():
    """
    Entry webhook called by Twilio when a call arrives.
    Responds with a Gather (speech) to collect caller's voice.
    The 'action' attribute points to /twilio/handle_gather which will process SpeechResult.
    """
    resp = VoiceResponse()
    gather = Gather(input="speech", action=url_for('twilio_handle_gather', _external=True),
                    method="POST", timeout=5, speech_timeout="auto")
    # Prompt for the caller
    gather.say(TWILIO_LISTEN_PROMPT)
    resp.append(gather)
    # If gather fails or times out, say goodbye
    resp.say("I didn't hear anything. Goodbye.")
    return Response(str(resp), mimetype="application/xml")


@app.route("/twilio/handle_gather", methods=["GET", "POST"])
def twilio_handle_gather():
    """
    Twilio will call this after the Gather completes with 'SpeechResult' parameter.
    We forward the transcript to the main app via Socket.IO and wait briefly for assistant response.
    Then we reply to the caller with TwiML <Say>.
    """
    speech_result = request.values.get("SpeechResult", "").strip()
    confidence = request.values.get("Confidence", "")
    from_number = request.values.get("From", "")
    log.info("Twilio Gather returned SpeechResult=%s (confidence=%s) from=%s", speech_result, confidence, from_number)

    resp = VoiceResponse()

    if not speech_result:
        resp.say("Sorry, I did not catch that. Please try again later. Goodbye.")
        return Response(str(resp), mimetype="application/xml")

    # Forward to main app via Socket.IO and await response
    bridge = AssistantBridge(MAIN_APP_SOCKETIO_URL)
    assistant_text = bridge.send_and_wait_response(speech_result, timeout=ASSISTANT_RESPONSE_TIMEOUT)

    if assistant_text:
        # Reply using Say. Keep message reasonably short; Twilio will synthesize voice.
        resp.say(assistant_text)
    else:
        # fallback reply if no response from agent
        resp.say("I'm sorry. I couldn't get an answer at the moment. A human will be notified. Goodbye.")

    return Response(str(resp), mimetype="application/xml")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 3000))
    log.info("Starting Twilio webhook server on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=False)
