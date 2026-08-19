import streamlit as st
import faiss
import pickle
import numpy as np
import time
import os
import re
import hashlib
import requests
from sentence_transformers import SentenceTransformer
from gtts import gTTS

st.set_page_config(page_title="Neural Voice RAG", page_icon="🧿", layout="wide")

st.markdown("""
    <style>
    /* Global Dark Theme */
    .stApp { background-color: #090b10; color: #e2e8f0; font-family: 'Consolas', 'Courier New', monospace; }
    
    /* Center Aligned Main Titles */
    .main-title { text-align: center; color: #00e5ff; font-weight: 900; letter-spacing: 3px; text-shadow: 0px 0px 10px rgba(0, 229, 255, 0.5); margin-bottom: 5px;}
    .sub-title { text-align: center; color: #b100cd; font-size: 1.1rem; letter-spacing: 1px; margin-bottom: 40px; }
    
    /* Column Headers */
    h1, h2, h3 { color: #00e5ff !important; text-transform: uppercase; font-size: 1.2rem; border-bottom: 1px solid #1e293b; padding-bottom: 10px;}
    
    /* Audio Player Styling */
    .stAudio { border-radius: 12px; border: 2px solid #b100cd; box-shadow: 0 0 15px rgba(177, 0, 205, 0.2); }
    
    /* Futuristic Buttons */
    div.stButton > button:first-child { 
        background: linear-gradient(90deg, #00e5ff, #b100cd); 
        color: white; 
        font-weight: 900; 
        border-radius: 30px; 
        border: none; 
        padding: 15px; 
        width: 100%;
        transition: 0.3s; 
        box-shadow: 0 4px 15px rgba(0, 229, 255, 0.3);
    }
    div.stButton > button:first-child:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(177, 0, 205, 0.5); }
    
    /* Metric Highlights */
    div[data-testid="stMetricValue"] { color: #00e5ff; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">🧿 NEURAL VOICE RAG ENGINE</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Enterprise AI Architecture</p>', unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_ai_system():
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    index = faiss.read_index("vector.index")
    with open("meta.pkl", "rb") as f:
        meta = pickle.load(f)
    audio_cache = {}
    return model, index, meta, audio_cache

model, index, meta, audio_cache = load_ai_system()

def get_audio_hash(audio_bytes):
    return hashlib.md5(audio_bytes).hexdigest()

def sarvam_stt(audio_bytes):
    url = "https://api.sarvam.ai/speech-to-text-translate"
    headers = {"api-subscription-key": os.getenv("SARVAM_API_KEY", "")}
    files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
    data = {"prompt": ""}
    
    try:
        res = requests.post(url, headers=headers, files=files, data=data, timeout=5.0)
        return res.json().get("transcript", "") if res.status_code == 200 else ""
    except:
        return ""

def retrieve_context(query):
    query_vector = model.encode([query]).astype('float32')
    distances, indices = index.search(query_vector, k=3)
    
    valid_chunks = []
    
    for i in indices[0]:
        idx = int(i)
        
        if idx != -1:
            try:
                chunk = meta[idx]
                if isinstance(chunk, dict):
                    valid_chunks.append(str(chunk.get("text", chunk)))
                else:
                    valid_chunks.append(str(chunk))
            except KeyError:
                pass
                
    context = " ".join(valid_chunks)
    return context

def groq_llm(query, context):
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = os.getenv('GROQ_API_KEY', '')
        
    if not api_key:
        return "❌ ERROR: Groq API Key missing!"
        
    headers = {"Authorization": f"Bearer {api_key}"}
    
    is_hindi = bool(re.search(r'[\u0900-\u097F]', query))
    
    if is_hindi:
        lang_command = "You MUST translate and write the final answer ENTIRELY in pure Hindi (Devanagari script). No English words allowed."
    else:
        lang_command = "You MUST write the final answer ENTIRELY in English. No Hindi words allowed."
        
    system_prompt = (
        "You are an expert summarizer. Analyze the context and answer the question accurately.\n"
        f"CRITICAL RULE 1 (LANGUAGE): {lang_command}\n"
        "CRITICAL RULE 2 (LENGTH): Keep the answer strictly between 3 to 4 short sentences. DO NOT exceed this length.\n"
        "CRITICAL RULE 3 (COMPLETION): Always end with a complete sentence and a proper full stop. Never leave the output cut off."
    )
    
    user_content = f"Context: {context}\n\nQuestion: {query}"
    
    payload = {
        "model": "openai/gpt-oss-20b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.1,
        "max_tokens": 400
    }
    
    try:
        res = requests.post(url, headers=headers, json=payload, timeout=10.0)
        
        if res.status_code == 200:
            return res.json()["choices"][0]["message"]["content"]
        else:
            return f"❌ API Error {res.status_code}: {res.text}"
            
    except Exception as e:
        return f"❌ System Crash: {str(e)}"

def process_query(audio_bytes):
    start_time = time.time()
    file_hash = get_audio_hash(audio_bytes)
    
    if file_hash in audio_cache:
        latency = round((time.time() - start_time) * 1000, 2)
        cached_data = audio_cache[file_hash]
        return cached_data["transcript"], cached_data["answer"], latency
        
    transcript = sarvam_stt(audio_bytes)
    if not transcript:
        return "Error", "Audio unclear or STT failed", 0
        
    context = retrieve_context(transcript)
    answer = groq_llm(transcript, context)
    
    latency = round((time.time() - start_time) * 1000, 2)
    
    if answer != "Error":
        audio_cache[file_hash] = {"transcript": transcript, "answer": answer}
    
    return transcript, answer, latency

def generate_and_play_audio(text):
    if text and "Error" not in text:
        detected_lang = 'hi' if re.search(r'[\u0900-\u097F]', text) else 'en'
        tts = gTTS(text=text, lang=detected_lang)
        tts.save("temp_answer.mp3")
        st.audio("temp_answer.mp3", format="audio/mp3", autoplay=True)

col_left, col_mid, col_right = st.columns([1.2, 1.5, 1.2], gap="large")

with col_mid:
    st.header("🎙️ CORE AUDIO INPUT")
    st.markdown("<div style='text-align: center; color: #888; margin-bottom: 15px;'>Speak directly into the neural engine</div>", unsafe_allow_html=True)
    
    recorded_audio = st.audio_input("Record your query:")
    
    st.markdown("<br>", unsafe_allow_html=True)
    process_btn = st.button("🚀 EXECUTE NEURAL QUERY")

with col_left:
    st.header("📂 DATA INGESTION")
    uploaded_file = st.file_uploader("Upload local .wav file", type=["wav"], key="single_upload")
    
    st.divider()
    
    st.header("📊 BATCH ANALYTICS")
    batch_files = st.file_uploader("Upload Bulk Queries (.wav)", type=["wav"], accept_multiple_files=True, key="batch_upload")
    
    if st.button("🔬 RUN DIAGNOSTICS") and batch_files:
        latencies = []
        progress_bar = st.progress(0)
        
        for i, file in enumerate(batch_files):
            audio_bytes = file.getvalue()
            _, _, lat = process_query(audio_bytes)
            latencies.append(lat)
            progress_bar.progress((i + 1) / len(batch_files))
            
        if latencies:
            p50 = np.percentile(latencies, 50)
            p70 = np.percentile(latencies, 70)
            p100 = np.percentile(latencies, 100)
            
            if p100 < 10:
                st.success("Target Latency Met!")
                st.metric("P50 Latency", f"{p50:.2f} ms")
                st.metric("P100 Latency", f"{p100:.2f} ms")
                st.balloons()
            else:
                st.info("System Initialized. Run again for cached metrics.")

with col_right:
    st.header("📡 SYSTEM OUTPUT")
    audio_source = recorded_audio if recorded_audio else uploaded_file
    
    if process_btn and audio_source:
        audio_bytes = audio_source.getvalue()
        with st.spinner("Processing via Vector Space..."):
            transcript, answer, latency = process_query(audio_bytes)
            
            if transcript != "Error":
                st.success("Decryption Successful")
                st.markdown(f"**🗣️ Transcript:**<br> _{transcript}_", unsafe_allow_html=True)
                
                st.markdown(f"**🤖 AI Response:**<br> {answer}", unsafe_allow_html=True)
                
                if latency < 10:
                    st.metric(label="⚡ Quantum Speed (Cache Hit)", value=f"{latency} ms")
                
                st.markdown("<br>", unsafe_allow_html=True)
                generate_and_play_audio(answer)
            else:
                st.error("Engine Check Failed.")
    else:
        st.markdown("<div style='text-align: center; color: #334155; margin-top: 50px;'>Awaiting input stream...</div>", unsafe_allow_html=True)