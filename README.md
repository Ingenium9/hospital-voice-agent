# 📌 Loop AI Hospital Network Voice Agent

This project implements a **voice-enabled AI assistant** that can search hospitals, verify network status, handle follow-up questions, and detect out-of-scope queries. It includes a clean web interface plus optional Twilio phone-call integration.

This submission fulfills the requirements of the **Loop AI Voice Assignment** — including the two mandatory test queries.

---

## 🚀 Features

### 🎤 Voice Input (Browser Microphone)
- Real-time speech recognition  
- Continuous listening mode  
- Automatic greeting when conversation starts  
- WebSocket streaming via Socket.IO  

### 🧠 AI Intent Understanding
- Extracts search terms (hospital name, city, intent)  
- Detects search type:  
  - List hospitals  
  - Network verification  
  - General inquiry  
- Handles follow-up clarifications  
- Asks questions when user query is incomplete  

### 🔍 Hospital Search (RAG + Exact Match)
- FAISS vector database for semantic search  
- Exact match search using Pandas  
- Hybrid pipeline (exact → semantic → fallback)  

### 🏥 Network Verification
Handles queries like:  
> “Can you confirm if Manipal Sarjapur in Bangalore is in my network?”  

### 🗂️ Out-of-Scope Detection
If the user asks irrelevant questions:  
> “I’m sorry, I can’t help with that. I am forwarding this to a human agent.”  

Conversation then ends gracefully.

### 🔊 Voice Output (TTS)
- ElevenLabs TTS (if API key is available)  
- Server-side pyttsx3 fallback  
- Audio returned to browser in Base64  

### 🕸️ Web UI
- Clean, minimal UI  
- Centered microphone button  
- Listening visualizer  
- Conversation history  
- System-ready badge  

---

## 🛠️ Tech Stack

### Backend
- Python 3  
- Flask + Flask-SocketIO  
- FAISS (vector database)  
- LangChain embeddings (MiniLM-L6-v2)  
- ElevenLabs text-to-speech  
- SpeechRecognition + pydub  
- Gemini/OpenAI (optional, for query extraction)  

### Frontend
- HTML / CSS / JavaScript  
- Socket.IO client  
- Audio player + waveform animation  

---

## 📁 Project Structure

hospital-voice-agent/
│
├── app.py # Main server + WebSocket + Twilio webhook
├── requirements.txt
├── README.md
├── setup.py
│
├── templates/
│ └── index.html # Web interface
│
├── static/
│ ├── script.js # Frontend logic + audio + WebSocket client
│ └── style.css # UI styling
│
├── data/
│ └── hospitals.csv # Dataset
│
├── utils/
│ ├── data_loader.py # CSV loader + FAISS vector store
│ ├── conversation_manager.py # Conversation history manager
│ ├── rag_engine.py # RAG logic + LLM response generation
│ └── voice_handler.py # Mic input + ElevenLabs TTS
│
└── twilio_server.py # Optional Twilio phone-call integration


---

## ▶️ How to Run Locally

### 1. Install dependencies
pip install -r requirements.txt


### 2. Create a `.env` file
OPENAI_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
ELEVENLABS_API_KEY=your_key_here
FLASK_SECRET=your_flask_secret


All keys are optional — the app still works without TTS.

### 3. Start the server

### 4. Open the UI
Visit:

http://localhost:5000


Click the microphone icon and start talking.

---

## 📞 Twilio Integration (Optional Bonus)
The project includes a **Twilio webhook** (`/twilio/voice`) that can connect the Loop AI agent with an actual phone call through:

- A purchased Twilio phone number  
- A public URL via ngrok  
- TwiML-based call handling  

Users can speak to the same assistant over a phone call.

---

## 🧪 Required Test Cases (Both Implemented)

### 1️⃣ “Tell me 3 hospitals around Bangalore.”
The system performs:
- Vector search  
- Smart ranking  
- Natural response with address + network status  

### 2️⃣ “Can you confirm if Manipal Sarjapur in Bangalore is in my network?”
The system performs:
- Exact match search  
- Clear verification response  

---

## 🎥 Loom Video
A demo video showcasing:
- Starting a voice conversation  
- Asking both mandatory queries  
- Hearing the assistant’s voice responses  

---

## 📝 Notes
- The system automatically asks clarifying questions when needed.  
- Out-of-scope queries end the conversation politely.  
- Dataset loading and vector search are optimized for large CSV files.  
- Fully functional even without expensive APIs.  

---

## ⭐ Author
**Shrishail Rugge**  
Loop AI Assignment — Hospital Voice Assistant

---

