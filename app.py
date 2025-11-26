# app.py (full updated)
import os
import time
import traceback
import tempfile
import base64
import threading
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

# local utils
from utils.data_loader import HospitalDataLoader

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET', os.urandom(24))
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app)

# Globals
agent = None
connected_sid = None

# track which clients have already been greeted for the current session
greeted_sids = set()

# --------------------
# Optional server-side TTS (pyttsx3) - run in background thread to avoid "run loop already started"
# --------------------
try:
    import pyttsx3
    class TextToSpeech:
        def __init__(self):
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 150)
                self.tts_engine.setProperty('volume', 0.8)
            except Exception as e:
                print("pyttsx3 init failed:", e)
                self.tts_engine = None

        def _speak_blocking(self, text):
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e:
                print("pyttsx3 speak error (blocking):", e)

        def speak(self, text):
            if not self.tts_engine:
                return False
            try:
                # run in background thread to avoid blocking / run-loop conflicts
                t = threading.Thread(target=self._speak_blocking, args=(text,), daemon=True)
                t.start()
                return True
            except Exception as e:
                print("pyttsx3 speak error:", e)
                return False
except Exception:
    TextToSpeech = None

# --------------------
# ElevenLabs client wrapper (defensive)
# --------------------
eleven_client = None
eleven_usable = True  # flag to stop repeated failing attempts
ELEVEN_KEY = os.getenv('ELEVENLABS_API_KEY')

if ELEVEN_KEY:
    try:
        # try to import official client if installed
        from elevenlabs.client import ElevenLabs as _ElevenLabsClient
        try:
            eleven_client = _ElevenLabsClient(api_key=ELEVEN_KEY)
            print("ElevenLabs client initialized for TTS (client object present)")
        except Exception as e:
            print("ElevenLabs client init failed:", e)
            eleven_client = None
    except Exception as e:
        # fallback: elevenlabs package may have different structure or not installed
        print("ElevenLabs package import issue:", e)
        eleven_client = None
else:
    print("ELEVENLABS_API_KEY not set - assistant audio will be text-only or server-side TTS.")

def generate_tts_base64(text, voice="Rachel", _max_failures=3):
    """
    Attempt to generate base64-encoded audio for `text` using ElevenLabs client if available.
    Defensive: tries several method names and returns base64 or None.
    """
    global eleven_client, eleven_usable
    if not ELEVEN_KEY or not eleven_client or not eleven_usable:
        return None

    attempts = [
        ("generate", lambda client: client.generate(text=text, voice=voice, model="eleven_monolingual_v1")),
        ("text_to_speech", lambda client: client.text_to_speech(text=text, voice=voice)),
        ("synthesize", lambda client: client.synthesize(text=text, voice=voice)),
        ("tts", lambda client: client.tts(text=text, voice=voice)),
    ]

    last_exc = None
    for name, fn in attempts:
        if not hasattr(eleven_client, name):
            continue
        try:
            result = fn(eleven_client)
            audio_bytes = b""
            if result is None:
                continue
            if isinstance(result, (bytes, bytearray)):
                audio_bytes = bytes(result)
            else:
                try:
                    for chunk in result:
                        if isinstance(chunk, str):
                            chunk = chunk.encode('utf-8')
                        audio_bytes += chunk
                except TypeError:
                    try:
                        audio_bytes = bytes(result)
                    except Exception:
                        audio_bytes = b""
            if audio_bytes:
                return base64.b64encode(audio_bytes).decode("utf-8")
        except Exception as e:
            last_exc = e
            print(f"ElevenLabs TTS attempt via '{name}' failed:", e)
            continue

    print("All ElevenLabs TTS attempts failed. Last error:", last_exc)
    eleven_usable = False
    return None

# --------------------
# Gemini wrapper (defensive)
# --------------------
gemini_disabled = False
try:
    import google.generativeai as genai
except Exception:
    genai = None

class GeminiClient:
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("GEMINI_API_KEY missing")
        if not genai:
            raise RuntimeError("google.generativeai package not available")
        genai.configure(api_key=api_key)
        self.model = None

    def extract_search_terms(self, query):
        global gemini_disabled
        if gemini_disabled:
            return self.fallback_extraction(query)
        try:
            prompt = f"Extract concise search terms from: \"{query}\". Return a short phrase suitable for searching (no extra text)."
            resp = genai.generate(prompt=prompt)
            text = getattr(resp, "text", None) or (resp and resp[0] and getattr(resp[0], "text", None)) or str(resp)
            if isinstance(text, str):
                return text.strip()
            return str(text).strip()
        except Exception as e:
            print("Gemini extraction error:", e)
            gemini_disabled = True
            return self.fallback_extraction(query)

    def fallback_extraction(self, query):
        query_lower = query.lower()
        locations = ['bangalore', 'bengaluru', 'mumbai', 'delhi', 'chennai', 'kolkata', 'hyderabad', 'pune']
        found_location = next((loc for loc in locations if loc in query_lower), None)
        hospital_types = {
            'heart': 'cardiac', 'cardiac': 'cardiac', 'eye': 'eye', 'dental': 'dental',
            'children': 'children', 'maternity': 'maternity', 'emergency': 'emergency',
            'general': 'general', 'multi': 'multi', 'specialty': 'specialty'
        }
        found_type = next((ht for key, ht in hospital_types.items() if key in query_lower), 'hospital')
        if found_location:
            return f"{found_type} hospital {found_location}"
        return f"{found_type} hospital"

# --------------------
# Agent implementation
# --------------------
class HospitalVoiceAgent:
    def __init__(self, enable_server_mic=False):
        print("Initializing HospitalVoiceAgent...")
        gemini_key = os.getenv('GEMINI_API_KEY')
        try:
            if gemini_key and genai:
                self.llm_client = GeminiClient(gemini_key)
                print("Gemini client initialized")
            else:
                self.llm_client = None
                if not genai and gemini_key:
                    print("genai package not available, Gemini disabled.")
                else:
                    print("GEMINI_API_KEY not set; will use fallback extraction.")
        except Exception as e:
            print("Gemini init error:", e)
            self.llm_client = None

        # Data + vector store
        self.data_loader = HospitalDataLoader(data_file="data/hospitals.csv")
        df = self.data_loader.load_data()
        documents = self.data_loader.create_documents(df)
        self.vector_store = None
        try:
            self.vector_store = self.data_loader.create_vector_store(documents)
        except Exception as e:
            print("Vector store creation error:", e)

        self.tts = TextToSpeech() if TextToSpeech else None

        self.enable_server_mic = enable_server_mic
        if self.enable_server_mic:
            try:
                import speech_recognition as sr
                self.recognizer = sr.Recognizer()
                self.microphone = sr.Microphone()
                with self.microphone as s:
                    self.recognizer.adjust_for_ambient_noise(s, duration=1)
                print("Server microphone calibrated")
            except Exception as e:
                print("Server mic init failed:", e)
                self.recognizer = None
                self.microphone = None
        else:
            self.recognizer = None
            self.microphone = None

        self.is_listening = False
        print("Agent initialized")

    def extract_search_terms(self, query):
        if self.llm_client:
            try:
                return self.llm_client.extract_search_terms(query)
            except Exception:
                return self.llm_client.fallback_extraction(query)
        # fallback if no LLM
        return "hospitals " + query

    def is_out_of_scope(self, query):
        """
        Basic out-of-scope detector:
        - If query contains obvious unrelated keywords OR
        - If the query is a general chit-chat (contains 'weather', 'joke', 'how are you', 'openai pricing', etc.)
        This is intentionally simple and conservative. Adjust list as needed.
        """
        q = (query or "").lower()
        if not q.strip():
            return True
        unrelated_keywords = [
            "weather", "joke", "time", "date", "translate", "news", "play music",
            "stock", "price", "profit", "programming", "how are you", "who are you",
            "openai", "google", "chatgpt", "wikipedia", "movie", "sports", "recipe"
        ]
        for kw in unrelated_keywords:
            if kw in q:
                return True
        # if the sentence doesn't mention hospitals, hospital-like words, clinics, or locations, treat as out-of-scope
        allowed_tokens = ["hospital", "hospitals", "clinic", "hospitals around", "in my network", "network", "address", "near", "nearby", "city", "in"]
        if any(tok in q for tok in allowed_tokens):
            return False
        # check for city words
        cities = ['bangalore', 'bengaluru', 'mumbai', 'delhi', 'chennai', 'kolkata', 'hyderabad', 'pune']
        if any(c in q for c in cities):
            return False
        # default: out-of-scope
        return True

    def search_hospitals(self, query, k=3):
        """
        Improved search:
        - use extract_search_terms(query) normally
        - if returned terms are too generic (e.g., 'hospital' or 'hospitals'),
          use the raw query or perform an exact-match search first
        - returns (hospitals[], search_terms_used)
        """
        try:
            search_terms = self.extract_search_terms(query)
            st_lower = (search_terms or "").strip().lower()
            generic_tokens = {"hospital", "hospitals", "clinic", "healthcare", "medical"}
            is_generic = (st_lower in generic_tokens) or (st_lower == "") or (any(tok in st_lower for tok in generic_tokens) and len(st_lower.split()) <= 2)

            q_clean = (query or "").strip()
            looks_like_name = ("hospital" in q_clean.lower() and len(q_clean.split()) <= 5) or (q_clean.istitle() and len(q_clean.split()) <= 5)

            results = []
            # exact-match attempt if looks like a name or returned search_terms is generic
            if (is_generic or looks_like_name) and self.data_loader:
                try:
                    df_exact = self.data_loader.exact_match_search(hospital_name=q_clean, city=None)
                    if df_exact is not None and not df_exact.empty:
                        for _, r in df_exact.iterrows():
                            hospitals_entry = {
                                'name': r.get('hospital_name', 'N/A'),
                                'address': r.get('address', 'N/A'),
                                'city': r.get('city', 'N/A'),
                                'network_status': r.get('network_status', 'N/A')
                            }
                            results.append(hospitals_entry)
                        if results:
                            return results, q_clean
                except Exception:
                    # ignore exact-match failure and continue to vector search
                    pass

            # fallback to vector similarity search using a useful search term:
            use_for_vector = query if is_generic else search_terms
            if not use_for_vector or (isinstance(use_for_vector, str) and use_for_vector.strip() == ""):
                use_for_vector = query

            if self.vector_store:
                vs_results = self.vector_store.similarity_search(str(use_for_vector), k=k)
                for doc in vs_results:
                    m = getattr(doc, "metadata", {}) or {}
                    results.append({
                        'name': m.get('hospital_name', 'N/A'),
                        'address': m.get('address', 'N/A'),
                        'city': m.get('city', 'N/A'),
                        'network_status': m.get('network_status', 'N/A')
                    })
            return results, (use_for_vector or search_terms)
        except Exception as e:
            print("search_hospitals error:", e)
            return [], query

    def generate_response(self, query, hospitals, search_terms):
        """
        Improved, human-friendly response generation:
        - produce a numbered, readable list (1., 2., 3.) with address and network status on separate lines
        """
        try:
            if not search_terms:
                search_terms = query or ""
            s = str(search_terms).strip()
            # collapse duplicate adjacent words: "hospital hospital" -> "hospital"
            s_parts = s.split()
            compact_parts = []
            for w in s_parts:
                if compact_parts and compact_parts[-1].lower() == w.lower():
                    continue
                compact_parts.append(w)
            s_clean = " ".join(compact_parts).strip()

            # try to infer city if present at end
            city = None
            tokens = s_clean.lower().split()
            known_cities = ['bangalore', 'bengaluru', 'mumbai', 'delhi', 'chennai', 'kolkata', 'hyderabad', 'pune']
            for kc in known_cities:
                if kc in tokens:
                    city = kc.title()
                    break

            if not hospitals:
                if city:
                    return f"I couldn't find any hospitals around {city}. Try a different area or keyword."
                return "I couldn't find any hospitals matching your search. Try a different keyword or location."

            # Build lead sentence
            if city:
                lead = f"I found {len(hospitals)} hospitals around {city}:"
            else:
                # show friendly phrase "around <city>" only if city available, else generic
                lead = f"I found {len(hospitals)} hospitals for '{s_clean}':"

            lines = [lead]
            for i, h in enumerate(hospitals, start=1):
                name = h.get('name', 'Unknown')
                address = h.get('address', '').strip()
                hcity = h.get('city', '')
                network = h.get('network_status', 'Unknown')

                addr_display = address
                if addr_display and hcity and hcity.lower() not in addr_display.lower():
                    addr_display = f"{addr_display}, {hcity}"
                elif not addr_display and hcity:
                    addr_display = hcity

                entry_lines = [f"{i}. {name}"]
                if addr_display:
                    entry_lines.append(f"   Address: {addr_display}")
                entry_lines.append(f"   Network: {network}")
                lines.append("\n".join(entry_lines))

            response = "\n\n".join(lines)
            return response

        except Exception as e:
            print("generate_response error:", e)
            return "Sorry — I had trouble forming the answer."

    def process_voice_input(self, timeout=10, phrase_time_limit=10):
        if not self.recognizer or not self.microphone:
            return None
        try:
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            text = self.recognizer.recognize_google(audio)
            print("Server-mic user:", text)
            return text
        except Exception as e:
            print("process_voice_input error:", e)
            return None

# ---- Initialize agent ----
def initialize_agent():
    global agent
    try:
        enable_server_mic = os.getenv('ENABLE_SERVER_MIC', 'false').lower() == 'true'
        agent = HospitalVoiceAgent(enable_server_mic=enable_server_mic)
        return True
    except Exception as e:
        print("initialize_agent failed:", e)
        traceback.print_exc()
        agent = None
        return False

# ---- Routes & transcription ----
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "message": "Hospital Voice Agent running"})

@app.route('/transcribe', methods=['POST'])
def transcribe_route():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "no file uploaded"}), 400
    filename = secure_filename(file.filename or 'upload.webm')
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1] or ".webm")
    file.save(tf.name)
    tf.close()

    # try local transcription via pydub + speech_recognition
    try:
        from pydub import AudioSegment
        import speech_recognition as sr
        wav_path = tf.name + ".wav"
        AudioSegment.from_file(tf.name).export(wav_path, format="wav")
        r = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio = r.record(source)
            text_result = r.recognize_google(audio)
            print("Transcribed (google):", text_result)
            return jsonify({"text": text_result})
    except Exception as e:
        print("Local transcription failed:", e)

    # optional OpenAI whisper fallback
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        try:
            import openai
            openai.api_key = openai_key
            with open(tf.name, "rb") as f:
                res = openai.Audio.transcribe("gpt-4o-transcribe", f)
                text_result = res.get("text", "")
                print("Transcribed (OpenAI):", text_result)
                return jsonify({"text": text_result})
        except Exception as e:
            print("OpenAI transcription failed:", e)

    return jsonify({"text": ""}), 500

# ---- Socket handlers ----
@socketio.on('connect')
def handle_connect():
    global connected_sid, agent
    sid = request.sid
    connected_sid = sid
    print(f"Client connected: {sid}")
    if agent:
        agent.is_listening = False
    emit('connected', {'status': 'ready'})
    emit('system_status', {'initialized': agent is not None})

@socketio.on('disconnect')
def handle_disconnect():
    global connected_sid, agent, greeted_sids
    sid = request.sid
    print(f"Client disconnected: {sid}")
    if agent:
        agent.is_listening = False
    if sid in greeted_sids:
        greeted_sids.discard(sid)
    if connected_sid == sid:
        connected_sid = None

def _emit_assistant_response(sid, text):
    """
    Emit assistant response with TTS audio if available.
    """
    try:
        audio_b64 = generate_tts_base64(text) if eleven_usable else None
    except Exception as e:
        print("TTS generation wrapper error:", e)
        audio_b64 = None

    socketio.emit('assistant_response', {'text': text, 'audio': audio_b64}, to=sid)

@socketio.on('toggle_conversation')
def handle_toggle_conversation(data):
    action = (data or {}).get('action')
    sid = request.sid
    if action == 'start':
        handle_start_listening_internal(sid)
    elif action == 'stop':
        handle_stop_listening_internal(sid)
    else:
        emit('error', {'message': 'Unknown action for toggle_conversation'})

def handle_start_listening_internal(sid):
    global agent, greeted_sids
    if not agent:
        emit('error', {'message': 'Agent not initialized'}, to=sid)
        return

    agent.is_listening = False
    try:
        agent.is_listening = True
        socketio.emit('listening_status', {'status': 'active'}, to=sid)

        # Send greeting only once per start -> dedupe by sid
        if sid not in greeted_sids:
            greeting = "Hello! I'm Loop AI Hospital Network Assistant. How can I help you today?"
            _emit_assistant_response(sid, greeting)
            greeted_sids.add(sid)
        else:
            print(f"Greeting already sent to sid {sid}; skipping duplicate.")

        socketio.start_background_task(target=continuous_listening, sid=sid)
        print("Started background listening task for sid:", sid)
    except Exception as e:
        print("Error starting listening:", e)
        agent.is_listening = False
        socketio.emit('error', {'message': str(e)}, to=sid)

def handle_stop_listening_internal(sid):
    global agent, greeted_sids
    if not agent:
        socketio.emit('error', {'message': 'Agent not initialized'}, to=sid)
        return

    try:
        agent.is_listening = False
        if sid in greeted_sids:
            greeted_sids.discard(sid)

        farewell = "Thank you for using our service. Goodbye!"
        _emit_assistant_response(sid, farewell)
        try:
            if agent.tts:
                agent.tts.speak(farewell)
        except Exception:
            pass
        socketio.emit('listening_status', {'status': 'inactive'}, to=sid)
        print("Stopped listening for sid:", sid)
    except Exception as e:
        print("Error stopping listening:", e)
        socketio.emit('error', {'message': str(e)}, to=sid)

@socketio.on('start_listening')
def handle_start_listening():
    handle_start_listening_internal(request.sid)

@socketio.on('stop_listening')
def handle_stop_listening():
    handle_stop_listening_internal(request.sid)

def continuous_listening(sid):
    global agent
    print("continuous_listening started for sid:", sid)
    while agent and agent.is_listening:
        try:
            user_input = agent.process_voice_input()
            if user_input:
                socketio.emit('user_input', {'text': user_input}, to=sid)

                # out-of-scope detection: politely end interaction if out of scope
                if agent.is_out_of_scope(user_input):
                    out_msg = "I'm sorry, I can't help with that. I am forwarding this to a human agent."
                    _emit_assistant_response(sid, out_msg)
                    try:
                        if agent.tts:
                            agent.tts.speak(out_msg)
                    except Exception:
                        pass
                    # stop the conversation for this client
                    agent.is_listening = False
                    socketio.emit('listening_status', {'status': 'inactive'}, to=sid)
                    return

                # quick exact-match check (network verification)
                q_lower = user_input.lower()
                if ((" in my network" in q_lower) or ("confirm if" in q_lower) or q_lower.strip().startswith("is ")):
                    hospital_name = None
                    city = None
                    if " in " in q_lower:
                        parts = user_input.lower().split(" in ")
                        hospital_name = parts[0].replace("can you confirm if", "").replace("confirm if", "").replace("is", "").strip()
                        city = parts[1].split()[0].strip().title() if parts[1] else None
                    if "manipal" in q_lower and "sarjapur" in q_lower:
                        hospital_name = "Manipal Sarjapur"
                        city = "Bangalore"
                    if hospital_name:
                        df_results = agent.data_loader.exact_match_search(hospital_name=hospital_name, city=city) if agent else None
                        if df_results is not None and not df_results.empty:
                            row = df_results.iloc[0]
                            network_status = row.get('network_status', None)
                            if not network_status:
                                response_text = f"I found {row.get('hospital_name')} in {row.get('city')}. Network status not provided."
                            else:
                                response_text = f"Yes — {row.get('hospital_name')} in {row.get('city')} is {network_status}."
                        else:
                            response_text = f"I couldn't find {hospital_name} {('in ' + (city or 'the requested city')) if city else ''} in our network."
                        _emit_assistant_response(sid, response_text)
                        continue

                hospitals, search_terms = ([], user_input)
                if agent:
                    hospitals, search_terms = agent.search_hospitals(user_input)
                    # if multiple exact-match results and user asked a named hospital -> ask clarifying question
                    if len(hospitals) > 1:
                        # try to detect if user input looked like a name
                        if ("hospital" in user_input.lower()) or user_input.strip().istitle():
                            name_hint = user_input.strip()
                            clarify = f"I have found several hospitals with this name. In which city are you looking for {name_hint}? Please say the city or the number of the hospital from the list."
                            _emit_assistant_response(sid, clarify)
                            try:
                                if agent.tts:
                                    agent.tts.speak(clarify)
                            except Exception:
                                pass
                            # also send the preliminary numbered list so user can pick
                            preview = agent.generate_response(user_input, hospitals, search_terms)
                            _emit_assistant_response(sid, preview)
                            continue

                    response_text = agent.generate_response(user_input, hospitals, search_terms)
                else:
                    response_text = "Agent not initialized."

                _emit_assistant_response(sid, response_text)
                try:
                    if agent and agent.tts:
                        agent.tts.speak(response_text)
                except Exception as e:
                    print("server TTS (response) error:", e)
            time.sleep(0.2)
        except Exception as e:
            print("Error in continuous_listening:", e)
            traceback.print_exc()
            try:
                socketio.emit('error', {'message': str(e)}, to=sid)
            except Exception:
                pass
            time.sleep(1)
    print("continuous_listening ended for sid:", sid)

@socketio.on('text_input')
def handle_text_input(data):
    global agent
    sid = request.sid
    try:
        user_input = (data or {}).get('text', '').strip()
        if not user_input:
            emit('error', {'message': 'Empty text input'}, to=sid)
            return
        socketio.emit('user_input', {'text': user_input}, to=sid)

        # out-of-scope detection
        if agent.is_out_of_scope(user_input):
            out_msg = "I'm sorry, I can't help with that. I am forwarding this to a human agent."
            _emit_assistant_response(sid, out_msg)
            try:
                if agent.tts:
                    agent.tts.speak(out_msg)
            except Exception:
                pass
            # stop conversation for this client
            agent.is_listening = False
            socketio.emit('listening_status', {'status': 'inactive'}, to=sid)
            return

        q_lower = user_input.lower()
        if ((" in my network" in q_lower) or ("confirm if" in q_lower) or q_lower.strip().startswith("is ")):
            hospital_name = None
            city = None
            if " in " in q_lower:
                parts = user_input.lower().split(" in ")
                hospital_name = parts[0].replace("can you confirm if", "").replace("confirm if", "").replace("is", "").strip()
                city = parts[1].split()[0].strip().title() if parts[1] else None
            if "manipal" in q_lower and "sarjapur" in q_lower:
                hospital_name = "Manipal Sarjapur"
                city = "Bangalore"
            if hospital_name:
                df_results = agent.data_loader.exact_match_search(hospital_name=hospital_name, city=city) if agent else None
                if df_results is not None and not df_results.empty:
                    row = df_results.iloc[0]
                    network_status = row.get('network_status', None)
                    if not network_status:
                        response_text = f"I found {row.get('hospital_name')} in {row.get('city')}. Network status not provided."
                    else:
                        response_text = f"Yes — {row.get('hospital_name')} in {row.get('city')} is {network_status}."
                else:
                    response_text = f"I couldn't find {hospital_name} {('in ' + (city or 'the requested city')) if city else ''} in our network."
                _emit_assistant_response(sid, response_text)
                try:
                    if agent and agent.tts:
                        agent.tts.speak(response_text)
                except Exception as e:
                    print("server TTS (text_input exact-match) error:", e)
                return

        hospitals, search_terms = ([], user_input)
        response_text = "Service not initialized."
        if agent:
            hospitals, search_terms = agent.search_hospitals(user_input)
            # clarifying question if multiple found for named input
            if len(hospitals) > 1:
                if ("hospital" in user_input.lower()) or user_input.strip().istitle():
                    name_hint = user_input.strip()
                    clarify = f"I have found several hospitals with this name. In which city are you looking for {name_hint}? Please reply with the city or the number of the hospital."
                    _emit_assistant_response(sid, clarify)
                    try:
                        if agent.tts:
                            agent.tts.speak(clarify)
                    except Exception:
                        pass
                    preview = agent.generate_response(user_input, hospitals, search_terms)
                    _emit_assistant_response(sid, preview)
                    return

            response_text = agent.generate_response(user_input, hospitals, search_terms)

        _emit_assistant_response(sid, response_text)
        try:
            if agent and agent.tts:
                agent.tts.speak(response_text)
        except Exception as e:
            print("server TTS (text_input response) error:", e)
    except Exception as e:
        print("Error processing text input:", e)
        traceback.print_exc()
        socketio.emit('error', {'message': str(e)}, to=sid)
    
from twilio.twiml.voice_response import VoiceResponse, Gather

@app.route("/twilio/voice", methods=["POST"])
def twilio_voice():
    resp = VoiceResponse()

    # 1. Greet
    resp.say("Hello, this is Loop AI Hospital Assistant. Please say your question after the beep.", voice="Polly.Salli")

    # 2. Gather user speech
    gather = Gather(
        input="speech",
        speechTimeout="auto",
        action="/twilio/process-speech",
        language="en-IN"
    )
    resp.append(gather)

    return str(resp)

@app.route("/twilio/process-speech", methods=["POST"])
def process_speech():
    user_text = request.form.get("SpeechResult", "")

    # If no speech detected
    if not user_text:
        resp = VoiceResponse()
        resp.say("Sorry, I didn't catch that. Please try again.")
        resp.redirect("/twilio/voice")
        return str(resp)

    # Process using existing agent logic
    response_text = "Please wait, processing your request."

    global agent
    if agent:
        hospitals, search_terms = agent.search_hospitals(user_text)
        response_text = agent.generate_response(user_text, hospitals, search_terms)

    resp = VoiceResponse()
    resp.say(response_text, voice="Polly.Salli")
    resp.pause(length=1)
    resp.redirect("/twilio/voice")  # continue loop
    return str(resp)



# ---- Run server ----
if __name__ == '__main__':
    print("Starting Hospital Voice Agent server...")
    ok = initialize_agent()
    print(f"Agent initialized: {ok}")
    try:
        print("Server listening at http://localhost:5000")
        
        socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    except Exception as e:
        print("Fatal error while running server:", e)
        traceback.print_exc()
