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

st.set_page_config(page_title="Voice Enabled RAG Project", page_icon="✨", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=0, viewport-fit=cover">
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
/* Global Settings */
html, body, .stApp { font-family: 'Outfit', sans-serif; }
/* Hide Default Elements */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}
/* Theme Toggle */
div[data-testid="stCheckbox"] { position: fixed; top: 20px; right: 30px; z-index: 999999; background: #000; padding: 8px 16px; border-radius: 8px; border: 1px solid #0ff; box-shadow: 0 0 10px rgba(0, 255, 255, 0.5); transition: all 0.3s ease; }
div[data-testid="stCheckbox"]:hover { background: #0ff; }
div[data-testid="stCheckbox"]:hover p { color: #000 !important; }
/* Dark Theme Default (Cyber Neon) */
.stApp { background: #050505; background-image: radial-gradient(circle at 20% 50%, rgba(0, 255, 255, 0.05), transparent 40%), radial-gradient(circle at 80% 30%, rgba(255, 0, 255, 0.05), transparent 40%); color: #fff; }
/* Main Container */
.main .block-container { padding-top: 3rem !important; max-width: 1200px; }
/* Hero Section */
.hero-section { text-align: center; margin-bottom: 4rem; margin-top: 1rem; }
.hero-section h1 { font-size: 4.5rem; font-weight: 800; margin-bottom: 0.5rem; color: #fff; text-shadow: 0 0 15px #0ff, 0 0 30px #f0f; letter-spacing: -1px; text-transform: uppercase; line-height: 1.1; }
.hero-section p { font-size: 1.1rem; color: #0ff; font-weight: 500; max-width: 600px; margin: 0 auto; letter-spacing: 3px; text-transform: uppercase; }
/* Column Cards */
[data-testid="column"] > div { background: rgba(10, 10, 10, 0.8); border: 1px solid rgba(0, 255, 255, 0.3); border-radius: 12px; padding: 32px; box-shadow: 0 0 20px rgba(0, 255, 255, 0.05); transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275); height: 100%; position: relative; overflow: hidden; }
[data-testid="column"] > div::before { content: ''; position: absolute; top: 0; left: -100%; width: 50%; height: 100%; background: linear-gradient(to right, transparent, rgba(0,255,255,0.1), transparent); transform: skewX(-20deg); transition: 0.5s; }
[data-testid="column"] > div:hover::before { left: 150%; }
[data-testid="column"] > div:hover { transform: translateY(-5px); border-color: rgba(255, 0, 255, 0.5); box-shadow: 0 0 30px rgba(255, 0, 255, 0.15); }
/* Typography */
h1, h2, h3, p, span, div { color: #fff; }
h3 { font-weight: 700; font-size: 1.6rem !important; margin-bottom: 1.5rem !important; padding-bottom: 1rem; border-bottom: 2px dashed rgba(0,255,255,0.2); display: flex; align-items: center; gap: 10px; letter-spacing: 1px; text-transform: uppercase; }
/* Buttons */
.stButton > button { background: transparent !important; color: #0ff !important; border: 2px solid #0ff !important; padding: 0.75rem 1.5rem !important; border-radius: 4px !important; font-weight: 600 !important; font-size: 1.1rem !important; letter-spacing: 2px; text-transform: uppercase; box-shadow: 0 0 10px rgba(0,255,255,0.2), inset 0 0 10px rgba(0,255,255,0.1) !important; transition: all 0.3s ease !important; width: 100%; margin-top: 1rem; }
.stButton > button:hover { background: #0ff !important; color: #000 !important; box-shadow: 0 0 20px rgba(0,255,255,0.6), inset 0 0 20px rgba(255,255,255,0.5) !important; transform: scale(1.02) !important; }
/* Secondary Buttons (Run Analytics) */
[data-testid="column"]:nth-child(2) .stButton > button { border-color: #f0f !important; color: #f0f !important; box-shadow: 0 0 10px rgba(255,0,255,0.2), inset 0 0 10px rgba(255,0,255,0.1) !important; }
[data-testid="column"]:nth-child(2) .stButton > button:hover { background: #f0f !important; color: #000 !important; box-shadow: 0 0 20px rgba(255,0,255,0.6) !important; }
/* Uploaders */
[data-testid="stFileUploader"] { background: rgba(0, 0, 0, 0.5); border: 2px dashed #0ff; border-radius: 8px; padding: 1rem; transition: all 0.3s ease; }
[data-testid="stFileUploader"]:hover { border-color: #f0f; background: rgba(20, 20, 20, 0.8); }
/* Audio Player */
.stAudio { border-radius: 8px; box-shadow: 0 0 15px rgba(0, 255, 255, 0.2); filter: hue-rotate(180deg) invert(1); }
/* Metrics */
[data-testid="stMetricValue"] { font-size: 2.5rem !important; font-weight: 700; color: #0ff; text-shadow: 0 0 10px rgba(0,255,255,0.5); }
[data-testid="stMetricLabel"] { font-size: 0.8rem !important; color: #f0f !important; font-weight: 600; text-transform: uppercase; letter-spacing: 2px; }
/* Alerts */
.stAlert { border-radius: 4px !important; border: 1px solid rgba(0,255,255,0.3) !important; background: rgba(0,0,0,0.8) !important; box-shadow: 4px 4px 0px rgba(0,255,255,0.3) !important; }
/* Mobile adjustments */
@media (max-width: 768px) { .main .block-container { padding: 2rem 1rem !important; } [data-testid="column"] > div { padding: 20px; margin-bottom: 20px; } div[data-testid="stCheckbox"] { top: 10px; right: 10px; padding: 6px 12px; } .hero-section h1 { font-size: 3rem; } }
</style>
""", unsafe_allow_html=True)

if 'is_light' not in st.session_state:
    st.session_state.is_light = False
is_light_mode = st.toggle("🌙 / ☀️", key="is_light")

if is_light_mode:
    st.markdown("""
<style>
/* Light Theme Default (Neumorphic) */
.stApp { background: #f0f0f3; background-image: none; color: #2d3436; }
h1, h2, h3, p, span, div { color: #2d3436; }
.hero-section h1 { color: #2d3436; text-shadow: 5px 5px 15px #cccdcf, -5px -5px 15px #ffffff; background: none; -webkit-text-fill-color: #2d3436; }
.hero-section p { color: #636e72; font-weight: 500; letter-spacing: 1px; }
[data-testid="column"] > div { background: #f0f0f3; border: none; border-radius: 30px; box-shadow: 20px 20px 60px #cccdcf, -20px -20px 60px #ffffff; transition: all 0.3s ease; }
[data-testid="column"] > div:hover { box-shadow: inset 10px 10px 30px #cccdcf, inset -10px -10px 30px #ffffff; transform: none; border: none; }
[data-testid="column"] > div::before { display: none; }
h3 { border-bottom: none; border-left: 5px solid #2d3436; padding-left: 15px; margin-left: -15px; }
[data-testid="stFileUploader"] { background: #f0f0f3; border: none; border-radius: 20px; box-shadow: inset 10px 10px 20px #cccdcf, inset -10px -10px 20px #ffffff; }
[data-testid="stFileUploader"]:hover { border: none; background: #f0f0f3; }
div[data-testid="stCheckbox"] { background: #f0f0f3; border: none; box-shadow: 5px 5px 15px #cccdcf, -5px -5px 15px #ffffff; color: #2d3436 !important; }
div[data-testid="stCheckbox"]:hover { background: #f0f0f3; box-shadow: inset 5px 5px 10px #cccdcf, inset -5px -5px 10px #ffffff; }
div[data-testid="stCheckbox"] p { color: #2d3436 !important; }
[data-testid="stMetricLabel"] { color: #636e72 !important; font-weight: 700; letter-spacing: 1px; }
[data-testid="stMetricValue"] { color: #2d3436; text-shadow: none; font-size: 2.5rem !important; }
.stButton > button { background: #f0f0f3 !important; color: #2d3436 !important; border: none !important; border-radius: 50px !important; box-shadow: 10px 10px 20px #cccdcf, -10px -10px 20px #ffffff !important; font-weight: 700 !important; transition: all 0.2s ease !important; }
.stButton > button:hover { background: #f0f0f3 !important; color: #00b894 !important; box-shadow: inset 10px 10px 20px #cccdcf, inset -10px -10px 20px #ffffff !important; transform: none !important; }
[data-testid="column"]:nth-child(2) .stButton > button:hover { color: #d63031 !important; }
.stAudio { filter: none; border-radius: 50px; box-shadow: 10px 10px 20px #cccdcf, -10px -10px 20px #ffffff; }
.stAlert { background: #f0f0f3 !important; border: none !important; box-shadow: 5px 5px 15px #cccdcf, -5px -5px 15px #ffffff !important; border-radius: 15px !important; }
</style>
""", unsafe_allow_html=True)



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
    url = "https://api.sarvam.ai/speech-to-text"
    headers = {"api-subscription-key": os.getenv("SARVAM_API_KEY", "")}
    files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
    data = {
        "model": "saaras:v3",
        "language_code": "unknown"
    }
    
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
    
    lang_command = (
        "Identify the exact language of the user's question (e.g., Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, etc.). "
        "You MUST reply entirely in that exact same language using its native script (e.g., Devanagari for Hindi, Bengali script for Bengali, etc.). "
        "If the question is in English, reply entirely in English. "
        "Never translate an Indian language query to English."
    )
        
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
        if re.search(r'[\u0980-\u09FF]', text): detected_lang = 'bn' # Bengali
        elif re.search(r'[\u0A00-\u0A7F]', text): detected_lang = 'pa' # Punjabi
        elif re.search(r'[\u0A80-\u0AFF]', text): detected_lang = 'gu' # Gujarati
        elif re.search(r'[\u0B80-\u0BFF]', text): detected_lang = 'ta' # Tamil
        elif re.search(r'[\u0C00-\u0C7F]', text): detected_lang = 'te' # Telugu
        elif re.search(r'[\u0C80-\u0CFF]', text): detected_lang = 'kn' # Kannada
        elif re.search(r'[\u0D00-\u0D7F]', text): detected_lang = 'ml' # Malayalam
        elif re.search(r'[\u0900-\u097F]', text): detected_lang = 'hi' # Hindi/Marathi
        else: detected_lang = 'en'
        
        tts = gTTS(text=text, lang=detected_lang)
        tts.save("temp_answer.mp3")
        
        st.audio("temp_answer.mp3", format="audio/mp3", autoplay=True)
        
        c1, c2 = st.columns([1, 1])
        
        with c1:
            with open("temp_answer.mp3", "rb") as f:
                st.download_button(
                    label="⬇️ Download Audio Answer",
                    data=f,
                    file_name="ai_response.mp3",
                    mime="audio/mp3",
                    use_container_width=True
                )
                
        with c2:
            import streamlit.components.v1 as components
            is_light = st.session_state.get("is_light", False)
            bg_col = "#ffffff" if is_light else "#1e293b"
            txt_col = "#0f172a" if is_light else "#f8fafc"
            border_col = "#cbd5e1" if is_light else "#334155"
            
            html_code = f"""
            <div style="display: flex; align-items: center; justify-content: center; gap: 12px; font-family: 'Inter', sans-serif; color: {txt_col}; height: 100%; margin-top: 8px;">
                <label style="font-weight: 500; font-size: 0.95rem; margin:0;">Voice Speed:</label>
                <select onchange="
                    var audios = window.parent.document.getElementsByTagName('audio');
                    if (audios.length > 0) {{
                        audios[audios.length - 1].playbackRate = parseFloat(this.value);
                    }}
                " style="background-color: {bg_col}; color: {txt_col}; border: 1px solid {border_col}; padding: 8px 12px; border-radius: 8px; cursor: pointer; width: 110px; font-weight: 500; font-size: 0.95rem; outline: none; transition: all 0.2s ease;">
                    <option value="0.5">0.5x</option>
                    <option value="0.75">0.75x</option>
                    <option value="1.0" selected>1.0x Normal</option>
                    <option value="1.25">1.25x</option>
                    <option value="1.5">1.5x</option>
                    <option value="2.0">2.0x</option>
                </select>
            </div>
            """
            components.html(html_code, height=60)

# Inject Custom X (Clear) Button globally without affecting column alignments
import streamlit.components.v1 as components
components.html("""
<script>
    setInterval(function() {
        var parent = window.parent.document;
        var audioInputs = parent.querySelectorAll('[data-testid="stAudioInput"]');
        
        audioInputs.forEach(function(audioInput) {
            if (!audioInput.querySelector('#custom-clear-mic')) {
                var x = parent.createElement('div');
                x.id = 'custom-clear-mic';
                x.innerHTML = '✖';
                x.style.position = 'absolute';
                x.style.right = '15px';
                x.style.top = '50%';
                x.style.transform = 'translateY(-50%)';
                x.style.cursor = 'pointer';
                x.style.fontSize = '13px';
                x.style.color = '#ff4b4b'; // Red color
                x.style.zIndex = '999';
                x.style.transition = 'transform 0.2s';
                x.title = "Clear Recording";
                x.style.display = 'none'; // Hidden by default
                
                x.onmouseover = function() { this.style.transform = 'translateY(-50%) scale(1.3)'; };
                x.onmouseout = function() { this.style.transform = 'translateY(-50%) scale(1)'; };
                
                // Attach to the actual record box (usually the second child after the label)
                var recordArea = audioInput.children.length > 1 ? audioInput.children[1] : audioInput;
                recordArea.style.position = 'relative';
                recordArea.appendChild(x);
                
                x.onclick = function(e) {
                    e.stopPropagation();
                    e.preventDefault();
                    var btns = audioInput.querySelectorAll('button');
                    btns.forEach(function(btn) {
                        var label = (btn.getAttribute('aria-label') || '').toLowerCase();
                        if (label.includes('clear') || label.includes('delete') || label.includes('remove') || label.includes('reset')) {
                            btn.click();
                        }
                    });
                };
            }
            
            // Toggle visibility based on whether audio is recorded (i.e. native clear button exists)
            var customX = audioInput.querySelector('#custom-clear-mic');
            if (customX) {
                var hasRecorded = false;
                var btns = audioInput.querySelectorAll('button');
                btns.forEach(function(btn) {
                    var label = (btn.getAttribute('aria-label') || '').toLowerCase();
                    if (label.includes('clear') || label.includes('delete') || label.includes('remove') || label.includes('reset')) {
                        hasRecorded = true;
                    }
                });
                customX.style.display = hasRecorded ? 'block' : 'none';
            }
            
            // Shift the timer text left so they don't overlap
            var allElements = audioInput.querySelectorAll('*');
            allElements.forEach(function(el) {
                if (el.childNodes.length === 1 && el.childNodes[0].nodeType === 3) {
                    if (/^\d{1,2}:\d{2}$/.test(el.innerText.trim())) {
                        if (el.style.transform !== 'translateX(-15px)') {
                            el.style.transform = 'translateX(-15px)';
                        }
                    }
                }
            });
        });
    }, 500);
</script>
""", height=0)

col_title, col2, col1 = st.columns([3, 3.5, 4.5], gap="medium")

with col_title:
    st.markdown("""
    <style>
    .hero-section { text-align: left !important; margin-top: 0 !important; margin-bottom: 2rem !important; }
    .hero-section h1 { font-size: 3.5rem !important; }
    .hero-section p { margin: 0 !important; }
    </style>
    <div class="hero-section">
        <h1>Voice Enabled RAG Project</h1>
        <p>Next-Generation Multilingual Voice Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

with col1:
    st.markdown("<h3>🎤 Voice Interaction</h3>", unsafe_allow_html=True)
    recorded_audio = st.audio_input("Record your question live:")

    st.markdown("<div style='text-align: center; color: #94a3b8; margin: 1.5rem 0; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 2px; font-weight: 500;'>or</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload an audio file (.wav)", type=["wav"], key="single_upload")
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")

    audio_source = recorded_audio if recorded_audio else uploaded_file
    
    if st.button("✨ Process Query", type="primary") and audio_source:
        audio_bytes = audio_source.getvalue()
        with st.spinner("Processing with AI..."):
            transcript, answer, latency = process_query(audio_bytes)
            
            if transcript != "Error":
                st.success("Query Processed Successfully!")
                st.markdown(f"**🗣️ Recognized Text:**\n> {transcript}")
                st.info(f"**🤖 AI Answer:**\n{answer}")
                
                if latency < 10:
                    st.metric(label="System Latency", value=f"{latency} ms")
                    st.toast('Ultra-Fast Latency Achieved!', icon='🚀')
                
                generate_and_play_audio(answer)
            else:
                st.error("Processing Failed. Please try again.")

with col2:
    st.markdown("<h3>⚡ Performance Analytics</h3>", unsafe_allow_html=True)
    
    benchmark_mic = st.audio_input("Record directly for Benchmark:", key="benchmark_mic")
    
    st.markdown("<div style='text-align: center; color: #94a3b8; margin: 1.5rem 0; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 2px; font-weight: 500;'>or</div>", unsafe_allow_html=True)
    
    batch_files = st.file_uploader("Upload Test Queries (.wav)", type=["wav"], accept_multiple_files=True, key="batch_upload")
    
    sources = batch_files if batch_files else ([benchmark_mic] if benchmark_mic else [])
    
    if st.button("📈 Run Analytics", type="secondary") and sources:
        latencies = []
        progress_bar = st.progress(0)
        
        st.write("Executing Benchmark Sequence...")
        for i, file in enumerate(sources):
            audio_bytes = file.getvalue()
            _, _, lat = process_query(audio_bytes)
            latencies.append(lat)
            st.write(f"File {i+1} Processed Successfully.")
            progress_bar.progress((i + 1) / len(sources))
            
        if latencies:
            p50 = np.percentile(latencies, 50)
            p70 = np.percentile(latencies, 70)
            p100 = np.percentile(latencies, 100)
            
            st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 2rem 0;'>", unsafe_allow_html=True)
            
            if p100 < 10:
                st.markdown("<h4 style='color: #34d399; margin-bottom: 1rem;'>🏆 Target Met (< 10ms)</h4>", unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                m1.metric("P50", f"{p50:.2f} ms")
                m2.metric("P70", f"{p70:.2f} ms")
                m3.metric("P100", f"{p100:.2f} ms")
                
                import streamlit.components.v1 as components
                components.html("""
                <script>
                    var parentWindow = window.parent;
                    if (!parentWindow.document.getElementById('confetti-script')) {
                        var script = parentWindow.document.createElement('script');
                        script.id = 'confetti-script';
                        script.src = 'https://cdn.jsdelivr.net/npm/canvas-confetti@1.6.0/dist/confetti.browser.min.js';
                        script.onload = function() { fireConfetti(); };
                        parentWindow.document.head.appendChild(script);
                    } else {
                        fireConfetti();
                    }
                    function shootRocket(x_pos) {
                        parentWindow.confetti({ particleCount: 25, angle: 90, spread: 15, startVelocity: 100, origin: { x: x_pos, y: 1 }, colors: ['#ffffff', '#ffaa00'], ticks: 80, zIndex: 999999 });
                        setTimeout(function() {
                            parentWindow.confetti({ particleCount: 150, angle: 90, spread: 360, startVelocity: 45, origin: { x: x_pos, y: 0.2 }, colors: ['#3b82f6', '#8b5cf6', '#ec4899', '#34d399', '#facc15', '#ef4444'], zIndex: 999999 });
                            setTimeout(function() {
                                parentWindow.confetti({ particleCount: 80, angle: 90, spread: 360, startVelocity: 35, origin: { x: x_pos, y: 0.2 }, colors: ['#ffffff'], zIndex: 999999 });
                            }, 150);
                        }, 500);
                    }
                    function fireConfetti() {
                        var rockets = 5;
                        var current = 0;
                        var interval = setInterval(function() {
                            if (current >= rockets) {
                                clearInterval(interval);
                                return;
                            }
                            var x_pos = 0.2 + (Math.random() * 0.6); // Random position across the screen
                            shootRocket(x_pos);
                            current++;
                        }, 600); // 1 rocket every 600ms
                    }
                </script>
                """, height=0)
