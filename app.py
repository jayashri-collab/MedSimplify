# -*- coding: utf-8 -*-
import streamlit as st
import os
import time
import re

# Import modules from our project
from preprocessing.clean_text import (
    extract_text_from_pdf, 
    clean_report_text, 
    extract_patient_metadata,
    ensure_sample_images
)
from utils.helper_functions import (
    get_medical_dictionary, 
    highlight_medical_terms, 
    generate_glossary, 
    format_report_download,
    highlight_report_headers
)
from model.inference import (
    run_simplification, 
    load_simplification_model, 
    run_risk_assessment,
    detect_medical_specialty,
    generate_lifestyle_recommendations,
    translate_text,
    analyze_medical_image
)
from PIL import Image

# Initialize sample medical images
ensure_sample_images()

# Helper function to strip HTML tags for SpeechSynthesis
def strip_html_tags(text: str) -> str:
    if not text:
        return ""
    # Remove HTML tags
    clean = re.sub(r'<[^>]+>', '', text)
    # Replace multiple spaces/newlines
    clean = re.sub(r'\s+', ' ', clean)
    return clean.strip()

# Helper to generate custom SpeechSynthesis buttons inside an iframe
def text_to_speech_html(text_to_read: str, language: str = "English") -> str:
    escaped_text = text_to_read.replace("\\", "\\\\").replace("`", "\\`").replace("'", "\\'").replace('"', '\\"').replace("\n", " ")
    
    # Map selection to standard voice codes
    lang_map = {
        "english": "en-US",
        "hindi": "hi-IN",
        "kannada": "kn-IN",
        "tamil": "ta-IN",
        "telugu": "te-IN"
    }
    lang_code = lang_map.get(language.lower(), "en-US")
    
    html_code = f"""
    <div style="display: flex; gap: 10px; margin-top: 12px; background: transparent;">
        <button id="play-btn" style="
            background-color: #38bdf8; 
            color: #0f172a; 
            border: none; 
            padding: 8px 16px; 
            border-radius: 6px; 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 0.85rem;
            font-weight: 600; 
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 6px;
            transition: all 0.2s ease-in-out;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        " onmouseover="this.style.backgroundColor='#7dd3fc'" onmouseout="this.style.backgroundColor='#38bdf8'">
            🔊 Listen
        </button>
        <button id="stop-btn" style="
            background-color: #ef4444; 
            color: #ffffff; 
            border: none; 
            padding: 8px 16px; 
            border-radius: 6px; 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 0.85rem;
            font-weight: 600; 
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 6px;
            transition: all 0.2s ease-in-out;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        " onmouseover="this.style.backgroundColor='#f87171'" onmouseout="this.style.backgroundColor='#ef4444'">
            ⏹ Stop
        </button>
    </div>
    
    <script>
        const playBtn = document.getElementById('play-btn');
        const stopBtn = document.getElementById('stop-btn');
        let utterance = null;
        
        playBtn.addEventListener('click', () => {{
            window.speechSynthesis.cancel();
            const textToSpeak = `{escaped_text}`;
            utterance = new SpeechSynthesisUtterance(textToSpeak);
            utterance.lang = '{lang_code}';
            
            // Look up localized voice if available
            const voices = window.speechSynthesis.getVoices();
            const targetLang = '{lang_code}'.toLowerCase();
            const targetPrefix = targetLang.split('-')[0];
            
            let matchingVoice = voices.find(v => v.lang.toLowerCase() === targetLang);
            if (!matchingVoice) {{
                matchingVoice = voices.find(v => v.lang.toLowerCase().startsWith(targetPrefix));
            }}
            
            if (matchingVoice) {{
                utterance.voice = matchingVoice;
            }}
            
            utterance.rate = 1.0;
            utterance.pitch = 1.0;
            window.speechSynthesis.speak(utterance);
        }});
        
        stopBtn.addEventListener('click', () => {{
            window.speechSynthesis.cancel();
        }});
    </script>
    """
    return html_code

# Page Configuration
st.set_page_config(
    page_title="MedSimplify - Patient-Friendly Medical Reports",
    page_icon="\U0001fa7a",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar Logo rendering (rendered first to stay at the top of the sidebar)
import base64
try:
    with open("assets/logo2.jpeg", "rb") as f:
        encoded_img = base64.b64encode(f.read()).decode()
    st.sidebar.markdown(
        f"""
        <div style="display: flex; flex-direction: column; align-items: center; text-align: center; margin-top: 15px; margin-bottom: 25px;">
            <img src="data:image/jpeg;base64,{encoded_img}" width="110" style="border-radius: 16px; box-shadow: 0 4px 15px rgba(20, 184, 166, 0.25); border: 2px solid rgba(20, 184, 166, 0.15); margin-bottom: 12px;">
            <h2 style="font-size: 1.35rem; font-weight: 700; margin: 0; background: linear-gradient(135deg, #14b8a6 0%, #3b82f6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">MedSimplify</h2>
            <p style="font-size: 0.8rem; color: var(--text-muted); margin: 4px 0 0 0;">AI Healthcare Assistant</p>
        </div>
        """,
        unsafe_allow_html=True
    )
except Exception:
    st.sidebar.image("assets/logo2.jpeg", width=110)
    st.sidebar.markdown("<div style='text-align: center; margin-bottom: 15px;'><b>MedSimplify</b><br><span style='font-size:0.8rem; color:var(--text-muted);'>AI Healthcare Assistant</span></div>", unsafe_allow_html=True)

# Theme Selector
theme_choice = st.sidebar.selectbox("🎨 UI Theme", ["Dark Mode", "Light Mode"], index=0)

# Custom Premium CSS Injection
if theme_choice == "Light Mode":
    root_vars = """
    :root {
        --bg-primary: #f8fafc;
        --card-bg: rgba(255, 255, 255, 0.9);
        --text-primary: #0f172a;
        --text-muted: #64748b;
        --border-color: rgba(0, 0, 0, 0.08);
        --sidebar-bg: #f1f5f9;
        --sidebar-border: #cbd5e1;
        --sidebar-text: #0f172a;
        --shadow-color: rgba(0, 0, 0, 0.08);
        --accent-primary: #0d9488;
        --accent-secondary: #2563eb;
        --term-badge-bg: #e2e8f0;
        --term-badge-border: #cbd5e1;
        --meta-card-bg: rgba(241, 245, 249, 0.85);
        --report-panel-bg: rgba(255, 255, 255, 0.95);
        --card-title-color: #0f172a;
        --hero-badge-bg: rgba(0, 0, 0, 0.04);
        --hero-badge-border: rgba(0, 0, 0, 0.08);
        --hero-badge-color: #0f172a;
        --subheading-color: #1e293b;
        --glow-shadow: rgba(13, 148, 136, 0.08);
        --glow-shadow-active: rgba(13, 148, 136, 0.15);
    }
    """
else:
    root_vars = """
    :root {
        --bg-primary: #111827;
        --card-bg: rgba(31, 41, 55, 0.85);
        --text-primary: #e2e8f0;
        --text-muted: #94a3b8;
        --border-color: rgba(255, 255, 255, 0.03);
        --sidebar-bg: #0f172a;
        --sidebar-border: #1f2937;
        --sidebar-text: #e2e8f0;
        --shadow-color: rgba(0, 0, 0, 0.25);
        --accent-primary: #14b8a6;
        --accent-secondary: #3b82f6;
        --term-badge-bg: #1e293b;
        --term-badge-border: #334155;
        --meta-card-bg: rgba(15, 23, 42, 0.6);
        --report-panel-bg: rgba(31, 41, 55, 0.85);
        --card-title-color: #e2e8f0;
        --hero-badge-bg: rgba(255, 255, 255, 0.04);
        --hero-badge-border: rgba(255, 255, 255, 0.08);
        --hero-badge-color: #e2e8f0;
        --subheading-color: #e2e8f0;
        --glow-shadow: rgba(20, 184, 166, 0.05);
        --glow-shadow-active: rgba(20, 184, 166, 0.2);
    }
    """

st.markdown("<style>" + root_vars + """
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    /* Global Font & Theme Overrides */
    html, body, [class*="css"], [data-testid="stAppViewContainer"] {
        font-family: 'Outfit', sans-serif !important;
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: var(--sidebar-bg) !important;
        border-right: 1px solid var(--sidebar-border) !important;
    }
    
    [data-testid="stSidebar"] * {
        color: var(--sidebar-text) !important;
    }
    
    /* Header (SaaS style hero section) */
    .hero-container {
        background: linear-gradient(135deg, rgba(20, 184, 166, 0.05) 0%, rgba(59, 130, 246, 0.05) 100%);
        border-radius: 20px;
        padding: 40px 30px;
        margin-bottom: 35px;
        box-shadow: 0 8px 32px 0 var(--shadow-color);
        border: 1px solid var(--border-color);
        backdrop-filter: blur(8px);
        text-align: left;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 10px 0;
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        font-size: 1.30rem;
        color: var(--text-muted);
        margin: 0 0 20px 0;
        font-weight: 400;
    }
    
    .hero-features {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin-top: 15px;
    }
    
    .hero-feature-badge {
        background-color: var(--hero-badge-bg);
        border: 1px solid var(--hero-badge-border);
        border-radius: 20px;
        padding: 6px 14px;
        font-size: 0.85rem;
        color: var(--hero-badge-color);
        display: flex;
        align-items: center;
        gap: 6px;
        transition: all 0.2s ease;
    }
    
    .hero-feature-badge:hover {
        background-color: rgba(20, 184, 166, 0.1);
        border-color: rgba(20, 184, 166, 0.3);
        transform: translateY(-2px);
    }

    /* Glassmorphism & SaaS Card Redesign */
    .glass-card {
        background: var(--card-bg) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-radius: 16px !important;
        padding: 24px !important;
        box-shadow: 0 8px 24px var(--shadow-color) !important;
        border: 1px solid var(--border-color) !important;
        margin-bottom: 20px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 30px var(--shadow-color) !important;
        border-color: rgba(20, 184, 166, 0.15) !important;
    }
    
    .card-title {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: var(--card-title-color) !important;
        margin-bottom: 16px !important;
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Highlighted terms styling */
    .med-highlight {
        background-color: rgba(20, 184, 166, 0.15);
        border-bottom: 2px dashed var(--accent-primary);
        color: var(--accent-primary);
        cursor: help;
        font-weight: 500;
        border-radius: 3px;
        padding: 0px 4px;
        transition: all 0.2s ease-in-out;
    }
    
    .med-highlight:hover {
        background-color: rgba(20, 184, 166, 0.3);
        color: var(--accent-primary);
        border-bottom-style: solid;
    }

    /* Term badge styling in Glossary */
    .term-badge {
        display: inline-block;
        background-color: var(--term-badge-bg);
        border: 1px solid var(--term-badge-border);
        border-radius: 16px;
        padding: 4px 12px;
        margin: 4px;
        font-size: 0.85rem;
        font-weight: 500;
        color: var(--text-primary);
        transition: all 0.2s ease;
    }
    
    .term-badge:hover {
        border-color: var(--accent-primary);
        background-color: rgba(20, 184, 166, 0.05);
    }
    
    .term-definition {
        color: var(--accent-primary);
        font-weight: 600;
    }
    
    /* Buttons Custom Styling */
    div.stButton > button {
        background-color: var(--accent-primary) !important;
        color: var(--bg-primary) !important;
        border: none !important;
        padding: 10px 24px !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 12px rgba(20, 184, 166, 0.2) !important;
        width: 100%;
    }
    
    div.stButton > button:hover {
        background-color: var(--accent-primary) !important;
        opacity: 0.9;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 16px rgba(20, 184, 166, 0.3) !important;
    }
    
    div.stButton > button:active {
        transform: translateY(1px) !important;
    }
    
    /* Secondary download button */
    div.stDownloadButton > button {
        background-color: transparent !important;
        color: var(--accent-primary) !important;
        border: 1px solid var(--accent-primary) !important;
        padding: 10px 24px !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease !important;
        width: 100%;
    }
    
    div.stDownloadButton > button:hover {
        background-color: rgba(20, 184, 166, 0.05) !important;
        border-color: var(--accent-primary) !important;
        color: var(--accent-primary) !important;
    }

    /* Specialty badges (Chips) */
    .specialty-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(20, 184, 166, 0.1);
        border: 1px solid rgba(20, 184, 166, 0.25);
        color: var(--accent-primary);
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
        box-shadow: 0 0 12px var(--glow-shadow);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from {
            box-shadow: 0 0 8px var(--glow-shadow);
        }
        to {
            box-shadow: 0 0 16px var(--glow-shadow-active);
        }
    }
    
    /* Modern mini-metadata cards */
    .meta-card {
        background: var(--meta-card-bg);
        border-radius: 12px;
        padding: 12px 16px;
        border: 1px solid var(--border-color);
        text-align: center;
        box-shadow: 0 4px 10px var(--shadow-color);
    }
    .meta-label {
        font-size: 0.75rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .meta-value {
        font-size: 0.95rem;
        color: var(--text-primary);
        font-weight: 600;
    }

    /* Risk score card custom styling */
    .risk-score-card {
        background: var(--card-bg);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 24px var(--shadow-color);
        border: 1px solid var(--border-color);
    }
    .risk-progress-container {
        display: flex;
        align-items: center;
        gap: 15px;
        margin: 15px 0;
    }
    .risk-bar {
        flex-grow: 1;
        height: 10px;
        background-color: var(--border-color);
        border-radius: 5px;
        overflow: hidden;
    }
    .risk-bar-fill {
        height: 100%;
        border-radius: 5px;
        transition: width 0.8s ease-in-out;
    }
    .risk-bar-high {
        background-color: #ef4444;
        box-shadow: 0 0 12px rgba(239, 68, 68, 0.4);
    }
    .risk-bar-mod {
        background-color: #f59e0b;
        box-shadow: 0 0 12px rgba(245, 158, 11, 0.4);
    }
    .risk-bar-low {
        background-color: #22c55e;
        box-shadow: 0 0 12px rgba(34, 197, 94, 0.4);
    }
    .risk-percentage {
        font-size: 1.15rem;
        font-weight: 700;
        min-width: 45px;
    }
    .risk-text-high { color: #ef4444 !important; font-weight: 700; }
    .risk-text-mod { color: #f59e0b !important; font-weight: 700; }
    .risk-text-low { color: #22c55e !important; font-weight: 700; }

    /* Report comparisons */
    .report-panel {
        background: var(--report-panel-bg);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 24px var(--shadow-color);
        border: 1px solid var(--border-color);
        height: 480px;
        overflow-y: auto;
    }
    
    /* Fade-in animation */
    .fade-in {
        animation: fadeIn 0.4s ease-out forwards;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(4px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# Main Premium Hero Header
st.markdown("""
<div class="hero-container">
    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 8px;">
        <span style="font-size: 3rem; filter: drop-shadow(0 2px 10px rgba(20, 184, 166, 0.3));">\U0001fa7a</span>
        <h1 class="hero-title" style="margin: 0; padding: 0; display: inline-block;">MedSimplify</h1>
    </div>
    <p class="hero-subtitle">AI-Powered Healthcare Assistant</p>
    <div class="hero-features">
        <span class="hero-feature-badge">\U0001f4c4 Medical Report Simplification</span>
        <span class="hero-feature-badge">\U0001f4f7 Medical Image Analysis</span>
        <span class="hero-feature-badge">\U0001f30d Multilingual Support</span>
        <span class="hero-feature-badge">\U0001f50a Voice Output</span>
        <span class="hero-feature-badge">\u26a0\ufe0f Risk Assessment</span>
    </div>
</div>
""", unsafe_allow_html=True)

# 11. Disclaimer at the top
st.warning("⚠️ **Disclaimer**: This application is intended for educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.")

# Sidebar for configuration and info
with st.sidebar:

    # Logo and theme selector are rendered at the top of the sidebar before CSS injection
    
    st.markdown("<p style='font-weight:600; text-transform:uppercase; font-size:0.8rem; color:var(--text-muted); letter-spacing:0.05em; margin-bottom: 10px;'>Settings</p>", unsafe_allow_html=True)
    
    # Selection of simplification model vs fallback
    simplification_mode = st.radio(
        "Translation Method",
        ("AI Model + Dictionary (Recommended)", "Dictionary Only (Fast Fallback)"),
        help="AI Model uses a neural network to rewrite sentences. Dictionary Only uses rule-based term replacements."
    )
    
    # 1. Output Language Selector
    output_language = st.selectbox(
        "🌐 Output Language",
        ("English", "Hindi", "Kannada", "Tamil", "Telugu"),
        index=0,
        help="Select the language for the final patient-friendly explanation."
    )
    
    st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
    st.markdown("<p style='font-weight:600; text-transform:uppercase; font-size:0.8rem; color:var(--text-muted); letter-spacing:0.05em; margin-bottom: 10px;'>Diagnostics</p>", unsafe_allow_html=True)
    
    # Check if GPU is available
    import torch
    gpu_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if gpu_available else "None (Using CPU)"
    
    st.markdown(f"**PyTorch CUDA Support:** {'Available' if gpu_available else 'Unavailable'}")
    st.markdown(f"**Computation Device:** {device_name}")
    
    # Pre-warm or show status of model loading in background
    if simplification_mode == "AI Model + Dictionary (Recommended)":
        with st.status("Verifying AI Model availability...", expanded=False) as status:
            try:
                model_pair = load_simplification_model()
                if status is not None:
                    status.update(label="AI Model loaded & ready", state="complete", expanded=False)
                st.write("Model loaded: `google/flan-t5-base` (~990MB)")
            except Exception as e:
                if status is not None:
                    status.update(label="Model failed to load. Will use Fallback.", state="error", expanded=True)
                st.error(f"Error details: {str(e)}")
    else:
        st.info("System is configured to run in fast Dictionary-Only mode.")

    st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
    st.markdown("<p style='font-weight:600; text-transform:uppercase; font-size:0.8rem; color:var(--text-muted); letter-spacing:0.05em; margin-bottom: 10px;'>About</p>", unsafe_allow_html=True)
    with st.expander("ℹ️ About MedSimplify", expanded=False):
        st.markdown("""
        MedSimplify is an AI-powered healthcare assistant designed to make medical summaries clear, accessible, and patient-friendly:
        
        * **Report Simplification**: Explains complex clinical text and provides localized glossary definitions.
        * **Multilingual Explanations**: Translates simplified summaries into Hindi, Kannada, Tamil, or Telugu.
        * **Clinical Risk Assessment**: Classifies overall health risk and flags critical conditions.
        * **X-Ray & Wound Analysis**: Uses computer vision to analyze medical scans and skin injuries.
        * **Camera Capture support**: Take wound photos directly to classify recovery/infection status.
        * **Speech Synthesis**: Synthesizes regional audio speech for simplified reports.
        """)

    # 12. Developer Footer
    st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
    st.markdown("<p style='font-weight:600; text-transform:uppercase; font-size:0.8rem; color:var(--text-muted); letter-spacing:0.05em; margin-bottom: 10px;'>Developers</p>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: var(--text-muted); font-size: 0.85rem; margin-top: 5px; background: var(--meta-card-bg); padding: 12px; border-radius: 12px; border: 1px solid var(--border-color);">
        <div style="margin-bottom: 12px;">
            <p style="margin-bottom: 0px; color: var(--accent-primary); font-weight: 700; font-size: 0.9rem;">Jayashri V. Hiremath</p>
            <p style="margin-bottom: 0px; font-size: 0.75rem;">B.Tech CSE</p>
        </div>
        <div>
            <p style="margin-bottom: 0px; color: var(--accent-primary); font-weight: 700; font-size: 0.9rem;">Tanishka Desai</p>
            <p style="margin-bottom: 0px; font-size: 0.75rem;">B.Tech CSI</p>
        </div>
        <p style="font-size: 0.75rem; margin-top: 8px; margin-bottom: 0; color: var(--text-muted);">Presidency University</p>
    </div>
    """, unsafe_allow_html=True)

# Create main navigation tabs
main_tab1, main_tab2 = st.tabs(["📄 Medical Report Analysis", "📷 X-Ray & Wound Analysis"])

with main_tab1:
    # File loading & inputs
    st.markdown("### 1. Select or Upload Medical Report")
    input_tab1, input_tab2, input_tab3 = st.tabs([
        "📤 Upload PDF Report", 
        "✍️ Paste Raw Text", 
        "📂 Load Sample Reports"
    ])

    report_text = ""

    # Tab 1: PDF Upload
    with input_tab1:
        uploaded_file = st.file_uploader(
            "Upload a medical report PDF file (e.g. Lab results, Discharge summaries, Radiology reports)", 
            type=["pdf"]
        )
        if uploaded_file is not None:
            try:
                with st.spinner("Extracting text from PDF..."):
                    raw_extracted = extract_text_from_pdf(uploaded_file)
                    report_text = clean_report_text(raw_extracted)
                    st.success(f"Successfully extracted text from: {uploaded_file.name}")
                    with st.expander("Preview Extracted Text"):
                        st.text(report_text[:1000] + ("..." if len(report_text) > 1000 else ""))
            except Exception as e:
                st.error(f"Error reading PDF: {e}")

    # Tab 2: Manual text paste
    with input_tab2:
        manual_input = st.text_area(
            "Paste the original medical report text below:",
            height=250,
            placeholder="Type or paste medical text here (e.g., Patient presented with acute dyspnea...)"
        )
        if manual_input:
            report_text = clean_report_text(manual_input)

    # Tab 3: Sample reports selector
    with input_tab3:
        st.info("Select a pre-loaded clinical report sample to see MedSimplify in action:")
        sample_choice = st.selectbox(
            "Choose a sample report",
            ("None selected", "Radiology Report (Chest X-Ray)", "Blood Test Report (CBC & Metabolic Panel)")
        )
        
        sample_files = {
            "Radiology Report (Chest X-Ray)": "sample_reports/sample_radiology.txt",
            "Blood Test Report (CBC & Metabolic Panel)": "sample_reports/sample_blood_test.txt"
        }
        
        if sample_choice != "None selected":
            file_path = sample_files[sample_choice]
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    report_text = f.read()
                st.success(f"Loaded {sample_choice} successfully.")
                with st.expander("View Selected Sample Text"):
                    st.text(report_text)
            else:
                st.error(f"Sample file not found at {file_path}")

    st.markdown("---")

    # Simplification trigger
    if st.button("✨ Simplify Report", type="primary", use_container_width=True):
        if not report_text.strip():
            st.warning("Please provide a medical report first (upload a PDF, paste text, or load a sample).")
        else:
            # Immersive loader sequence mimicking clinical analysis
            status_container = st.empty()
            with status_container.container():
                st.info("🧠 Analyzing Clinical Data...")
                time.sleep(0.4)
                st.info("📖 Simplifying Medical Terminology...")
                time.sleep(0.4)
                st.info("🌍 Generating Patient-Friendly Explanation...")
                time.sleep(0.3)
            status_container.empty()

            # Simplification execution
            start_time = time.time()
            
            # 1. Extract Patient Metadata (for UI display)
            metadata = extract_patient_metadata(report_text)
            
            # 5. Automatically detect medical specialty
            detected_specialty = detect_medical_specialty(report_text)
            
            # Run the inference pipeline
            simplified_text, log_msg, was_fallback = run_simplification(
                report_text, 
                method=simplification_mode
            )
                
            # 2. Output Language Translation
            translated_text = ""
            translation_error = False
            if output_language != "English":
                try:
                    translated_text = translate_text(simplified_text, output_language)
                except Exception as e:
                    translation_error = True
            
            # Run the risk assessment
            risk_level, risk_reason = run_risk_assessment(
                report_text,
                method=simplification_mode
            )
            # 3. Generate lifestyle recommendations
            lifestyle_recommendations = generate_lifestyle_recommendations(report_text)
            
            execution_time = time.time() - start_time
                
            # Display Execution status
            if was_fallback:
                st.warning(f"⚠️ {log_msg} (Time taken: {execution_time:.2f}s)")
            else:
                st.success(f"✅ {log_msg} (Time taken: {execution_time:.2f}s)")

            # Prepare highlighted text for UI rendering
            highlighted_original = highlight_report_headers(highlight_medical_terms(report_text))
            highlighted_simplified = highlight_medical_terms(simplified_text)
            
            # Generate glossary
            glossary = generate_glossary(report_text)
            
            # Display side-by-side comparison
            st.markdown("### 2. Analysis Results")
            
            # Specialty Display (round badge) and Patient Metadata grid (4 columns)
            specialty_emojis = {
                "Cardiology": "❤️ Cardiology",
                "Neurology": "🧠 Neurology",
                "Pulmonology": "🫁 Pulmonology",
                "Nephrology": "🧪 Nephrology",
                "Hematology": "🩸 Hematology",
                "Radiology": "🩻 Radiology",
                "General Medicine": "🩺 General Medicine"
            }
            specialty_display = specialty_emojis.get(detected_specialty, f"🏥 {detected_specialty}")
            
            st.markdown(f'<div class="specialty-badge" style="margin-bottom: 20px;">{specialty_display}</div>', unsafe_allow_html=True)
            
            meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
            with meta_col1:
                st.markdown(f"""
                <div class="meta-card">
                    <div class="meta-label">Patient Name</div>
                    <div class="meta-value">{metadata['name']}</div>
                </div>
                """, unsafe_allow_html=True)
            with meta_col2:
                st.markdown(f"""
                <div class="meta-card">
                    <div class="meta-label">Age</div>
                    <div class="meta-value">{metadata['age']}</div>
                </div>
                """, unsafe_allow_html=True)
            with meta_col3:
                st.markdown(f"""
                <div class="meta-card">
                    <div class="meta-label">Gender</div>
                    <div class="meta-value">{metadata['gender']}</div>
                </div>
                """, unsafe_allow_html=True)
            with meta_col4:
                st.markdown(f"""
                <div class="meta-card">
                    <div class="meta-label">Report Date</div>
                    <div class="meta-value">{metadata['date']}</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.write("") # Spacer

            # Split panels
            col1, col2 = st.columns(2)
            
            # Column 1: Original Report Card
            with col1:
                st.markdown(
    f"""
    <div class="report-panel fade-in">
        <div class="card-title">
            📄 Original Clinical Report
        </div>
        <div style="color: var(--text-primary); line-height: 1.7; white-space: pre-wrap; font-size: 0.95rem;">
            {highlighted_original}
        </div>
    </div>
    """, 
                    unsafe_allow_html=True
                )
                
            # Column 2: Simplified & Translated Report Card
            with col2:
                if output_language == "English":
                    st.markdown(
    f"""
    <div class="report-panel fade-in" style="border-color: var(--accent-primary) !important; border-width: 1.5px !important;">
        <div class="card-title" style="color: var(--accent-primary) !important;">
            ✨ Patient-Friendly Explanation
        </div>
        <div style="color: var(--text-primary); line-height: 1.7; white-space: pre-wrap; font-size: 0.95rem;">
            {highlighted_simplified}
        </div>
    </div>
    """, 
                        unsafe_allow_html=True
                    )
                    
                    # Voice Output synthesis component under the explanation card
                    st.components.v1.html(
                        text_to_speech_html(strip_html_tags(simplified_text), "English"),
                        height=65,
                        scrolling=False
                    )
                else:
                    if translation_error:
                        st.warning("⚠️ Translation unavailable. Showing English output.")
                        st.markdown(
    f"""
    <div class="report-panel fade-in" style="border-color: var(--accent-primary) !important; border-width: 1.5px !important;">
        <div class="card-title" style="color: var(--accent-primary) !important;">
            ✨ Patient-Friendly Explanation
        </div>
        <div style="color: var(--text-primary); line-height: 1.7; white-space: pre-wrap; font-size: 0.95rem;">
            {highlighted_simplified}
        </div>
    </div>
    """, 
                            unsafe_allow_html=True
                        )
                        
                        st.components.v1.html(
                            text_to_speech_html(strip_html_tags(simplified_text), "English"),
                            height=65,
                            scrolling=False
                        )
                    else:
                        st.markdown(
    f"""
    <div class="report-panel fade-in" style="height: 230px; margin-bottom: 20px;">
        <div class="card-title" style="color: var(--accent-primary) !important; margin-bottom: 8px !important;">
            ✨ Patient-Friendly Explanation (English)
        </div>
        <div style="color: var(--text-primary); line-height: 1.6; white-space: pre-wrap; font-size: 0.9rem;">
            {highlighted_simplified}
        </div>
    </div>
    """, 
                            unsafe_allow_html=True
                        )
                        
                        st.markdown(
    f"""
    <div class="report-panel fade-in" style="height: 230px; border-color: var(--accent-primary) !important; border-width: 1.5px !important;">
        <div class="card-title" style="color: var(--accent-primary) !important; margin-bottom: 8px !important;">
            🌍 Multilingual Patient Explanation ({output_language})
        </div>
        <div style="color: var(--text-primary); line-height: 1.6; white-space: pre-wrap; font-size: 0.9rem;">
            {translated_text}
        </div>
    </div>
    """, 
                            unsafe_allow_html=True
                        )
                        
                        # Voice Output for translation
                        st.components.v1.html(
                            text_to_speech_html(translated_text, output_language),
                            height=65,
                            scrolling=False
                        )
                
            # Clinical Risk Assessment Section
            st.markdown("### 3. Clinical Risk Assessment")
            if "High Risk" in risk_level:
                pct = 85
                bar_class = "risk-bar-high"
                text_class = "risk-text-high"
                icon = "🔴"
            elif "Moderate Risk" in risk_level:
                pct = 50
                bar_class = "risk-bar-mod"
                text_class = "risk-text-mod"
                icon = "🟡"
            else:
                pct = 15
                bar_class = "risk-bar-low"
                text_class = "risk-text-low"
                icon = "🟢"
                
            risk_bar_html = f"""
            <div class="risk-score-card fade-in" style="margin-bottom: 25px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                    <span style="font-size: 1.15rem; font-weight: 600; color: var(--text-primary);">Clinical Risk Score</span>
                    <span class="{text_class}" style="font-size: 1.25rem;">{icon} {risk_level}</span>
                </div>
                <div class="risk-progress-container">
                    <div class="risk-bar">
                        <div class="risk-bar-fill {bar_class}" style="width: {pct}%;"></div>
                    </div>
                    <span class="risk-percentage {text_class}">{pct}%</span>
                </div>
                <div style="color: var(--text-primary); font-size: 0.95rem; line-height: 1.6; margin-top: 10px;">
                    <strong>Assessment Findings:</strong> {risk_reason}
                </div>
            </div>
            """
            st.markdown(risk_bar_html, unsafe_allow_html=True)
                
            # Lifestyle Recommendations Section
            st.markdown("### 4. Lifestyle Recommendations")
            recs_list = "\n".join([f'<li style="margin-bottom: 8px; color: var(--text-primary); font-size: 0.95rem;">{rec}</li>' for rec in lifestyle_recommendations])
            st.markdown(f"""
            <div class="glass-card">
                <p style="color: var(--accent-primary); margin-top: 0px; font-weight: 600; font-size: 1.1rem; display: flex; align-items: center; gap: 8px;">🌱 Personalized Health Guidelines (Non-Prescriptive)</p>
                <ul style="padding-left: 20px; margin-bottom: 0px; margin-top: 10px;">
                    {recs_list}
                </ul>
            </div>
            """, unsafe_allow_html=True)
                
            # Glossary Section
            st.markdown("### 5. Patient's Glossary of Terms")
            if glossary:
                st.markdown(
    f"""
    <div class="glass-card" style="margin-bottom: 20px;">
        <p style="color: var(--text-muted); margin-top: 0px; font-size: 0.95rem; line-height: 1.6;">The following medical terms were identified in the report and defined in patient-friendly terms. Hover over these highlighted terms in the original and simplified cards above to display their definitions instantly.</p>
    </div>
    """,
                    unsafe_allow_html=True
                )
                
                # Render glossary items as badges
                for term, definition in sorted(glossary.items()):
                    st.markdown(
                        f'<span class="term-badge">{term.capitalize()} &rarr; <span class="term-definition">{definition}</span></span>',
                        unsafe_allow_html=True
                    )
            else:
                st.info("No specific complex medical terms from the dictionary were identified in this report.")

            # Download option
            st.markdown("---")
            download_doc = format_report_download(
                original=report_text,
                simplified=simplified_text,
                glossary=glossary,
                risk_level=risk_level,
                risk_reason=risk_reason,
                detected_specialty=detected_specialty,
                lifestyle_recommendations=lifestyle_recommendations,
                metadata=metadata,
                translated_explanation=translated_text,
                target_language=output_language
            )
            
            # Download button styled nicely
            st.download_button(
                label="📥 Download Simplified Report",
                data=download_doc,
                file_name="medsimplify_report.txt",
                mime="text/plain",
                use_container_width=True
            )

with main_tab2:
    st.markdown("### 📷 X-Ray & Wound Analysis")
    st.markdown("Upload an X-ray scan or wound photograph for zero-shot AI classification, confidence estimation, risk classification, and patient-friendly explanations.")
    
    img_select_tab1, img_select_tab2, img_select_tab3 = st.tabs([
    "📤 Upload Image",
    "📸 Camera Capture",
    "📂 Load Sample Images"
])
    pil_img = None
    
    with img_select_tab1:
        img_uploaded_file = st.file_uploader(
            "Upload a medical image (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            key="image_analysis_uploader"
        )
        if img_uploaded_file is not None:
            try:
                pil_img = Image.open(img_uploaded_file)
            except Exception as e:
                st.error(f"Error opening image: {str(e)}")
                pil_img = None

    with img_select_tab2:
        camera_image = st.camera_input(
            "Take a photo of the wound"
        )
        if camera_image:
            try:
                pil_img = Image.open(camera_image)
            except Exception as e:
                st.error(f"Error opening camera image: {str(e)}")
                pil_img = None
                
    with img_select_tab3:
        st.info("Select a pre-loaded mock scan or clinical photo:")
        sample_img_choice = st.selectbox(
            "Choose a sample image",
            ("None selected", "Sample Chest X-Ray (Lung Opacity)", "Sample Knee X-Ray (Femur Fracture)", "Sample Wound (Infected Lesion)")
        )
        
        sample_img_files = {
            "Sample Chest X-Ray (Lung Opacity)": "sample_reports/sample_chest_xray.png",
            "Sample Knee X-Ray (Femur Fracture)": "sample_reports/sample_knee_fracture.png",
            "Sample Wound (Infected Lesion)": "sample_reports/sample_wound.png"
        }
        
        if sample_img_choice != "None selected":
            sample_img_path = sample_img_files[sample_img_choice]
            try:
                pil_img = Image.open(sample_img_path)
            except Exception as e:
                st.error(f"Error loading sample image: {str(e)}")
                pil_img = None

    if pil_img is not None:
        # Display Preview & Trigger Column layout (Side-by-side Uploaded Image | AI Analysis)
        img_col1, img_col2 = st.columns([1, 1])
        with img_col1:
            st.markdown('<div class="glass-card" style="text-align: center;">', unsafe_allow_html=True)
            st.image(pil_img, use_container_width=True)
            st.markdown('<div style="font-size: 0.85rem; color: var(--text-muted); margin-top: 10px; font-weight: 500;">📷 Uploaded Scan / Photograph</div></div>', unsafe_allow_html=True)
        
        with img_col2:
            st.markdown("""
            <div class="glass-card">
                <div class="card-title">🔬 Image Diagnostic Center</div>
                <p style="color: var(--text-muted); font-size: 0.95rem; line-height: 1.6; margin: 0 0 15px 0;">
                    Our zero-shot neural classifiers will evaluate this image to identify anatomical type, run condition classifications, estimate diagnosis confidence, and run a clinical risk assessment.
                </p>
            </div>
            """, unsafe_allow_html=True)
            analyze_btn = st.button("🔍 Analyze Medical Image", type="primary", use_container_width=True)
            
        if analyze_btn:
            # Immersive loader sequence mimicking clinical analysis
            status_container = st.empty()
            with status_container.container():
                st.info("🧠 Analyzing Image Features...")
                time.sleep(0.4)
                st.info("🔍 Running Diagnostic Classifiers...")
                time.sleep(0.4)
                st.info("⚠️ Calculating Clinical Risk Score...")
                time.sleep(0.3)
            status_container.empty()
            
            try:
                start_img_time = time.time()
                
                # Run classification pipeline
                img_result = analyze_medical_image(pil_img)
                
                img_exec_time = time.time() - start_img_time
                st.success(f"✅ Image analyzed successfully in {img_exec_time:.2f}s")
                
                # Extract outputs
                det_type = img_result["image_type"]
                pred_cond = img_result["prediction"]
                conf_score = img_result["confidence"]
                img_risk = img_result["risk_level"]
                img_reason = img_result["risk_reason"]
                img_explanation = img_result["explanation"]
                
                # Handle translation
                translated_img_exp = ""
                if output_language != "English":
                    try:
                        translated_img_exp = translate_text(img_explanation, output_language)
                    except Exception as e:
                        st.warning("⚠️ Translation failed, displaying English output.")
                        translated_img_exp = ""
                
                # Display Results section
                st.markdown("### 2. Analysis Results")
                
                # Metric cards (4 columns grid)
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                with metric_col1:
                    st.markdown(f"""
                    <div class="meta-card">
                        <div class="meta-label">Image Type</div>
                        <div class="meta-value">📸 {det_type}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with metric_col2:
                    st.markdown(f"""
                    <div class="meta-card">
                        <div class="meta-label">AI Prediction</div>
                        <div class="meta-value">🔍 {pred_cond}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with metric_col3:
                    st.markdown(f"""
                    <div class="meta-card">
                        <div class="meta-label">Confidence</div>
                        <div class="meta-value">⚡ {conf_score*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                with metric_col4:
                    risk_emoji = "🟢"
                    if img_risk == "High Risk":
                        risk_emoji = "🔴"
                    elif img_risk == "Moderate Risk":
                        risk_emoji = "🟡"
                    st.markdown(f"""
                    <div class="meta-card">
                        <div class="meta-label">Risk Level</div>
                        <div class="meta-value">{risk_emoji} {img_risk}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.write("") # Spacer
                
                # Risk Assessment Score Bar Card
                st.markdown("### 3. Clinical Risk Assessment")
                
                if img_risk == "High Risk":
                    img_pct = 85
                    img_bar_class = "risk-bar-high"
                    img_text_class = "risk-text-high"
                    img_icon = "🔴"
                elif img_risk == "Moderate Risk":
                    img_pct = 50
                    img_bar_class = "risk-bar-mod"
                    img_text_class = "risk-text-mod"
                    img_icon = "🟡"
                else:
                    img_pct = 15
                    img_bar_class = "risk-bar-low"
                    img_text_class = "risk-text-low"
                    img_icon = "🟢"
                    
                img_risk_html = f"""
                <div class="risk-score-card fade-in" style="margin-bottom: 25px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                        <span style="font-size: 1.15rem; font-weight: 600; color: var(--text-primary);">Image Clinical Risk Score</span>
                        <span class="{img_text_class}" style="font-size: 1.25rem;">{img_icon} {img_risk}</span>
                    </div>
                    <div class="risk-progress-container">
                        <div class="risk-bar">
                            <div class="risk-bar-fill {img_bar_class}" style="width: {img_pct}%;"></div>
                        </div>
                        <span class="risk-percentage {img_text_class}">{img_pct}%</span>
                    </div>
                    <div style="color: var(--text-primary); font-size: 0.95rem; line-height: 1.6; margin-top: 10px;">
                        <strong>Assessment Findings:</strong> {img_reason}
                    </div>
                </div>
                """
                st.markdown(img_risk_html, unsafe_allow_html=True)
                    
                # Patient-Friendly Explanation (Glass Cards)
                st.markdown("### 4. Patient-Friendly Explanation")
                if output_language == "English" or not translated_img_exp:
                    st.markdown(f"""
                    <div class="report-panel fade-in" style="border-color: var(--accent-primary) !important; border-width: 1.5px !important; height: auto; max-height: none;">
                        <div class="card-title" style="color: var(--accent-primary) !important;">
                            ✨ Patient-Friendly Explanation
                        </div>
                        <div style="color: var(--text-primary); line-height: 1.7; font-size: 0.95rem;">
                            {img_explanation}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # TTS Output
                    st.components.v1.html(
                        text_to_speech_html(img_explanation, "English"),
                        height=65,
                        scrolling=False
                    )
                else:
                    st.markdown(f"""
                    <div class="report-panel fade-in" style="height: 200px; margin-bottom: 20px;">
                        <div class="card-title" style="color: var(--accent-primary) !important; margin-bottom: 8px !important;">
                            ✨ Patient-Friendly Explanation (English)
                        </div>
                        <div style="color: var(--text-primary); line-height: 1.6; font-size: 0.9rem;">
                            {img_explanation}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="report-panel fade-in" style="height: 200px; border-color: var(--accent-primary) !important; border-width: 1.5px !important;">
                        <div class="card-title" style="color: var(--accent-primary) !important; margin-bottom: 8px !important;">
                            🌍 Multilingual Patient Explanation ({output_language})
                        </div>
                        <div style="color: var(--text-primary); line-height: 1.6; font-size: 0.9rem;">
                            {translated_img_exp}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # TTS Output for translation
                    st.components.v1.html(
                        text_to_speech_html(translated_img_exp, output_language),
                        height=65,
                        scrolling=False
                    )
            except Exception as ex:
                st.error(f"Failed to run image analysis: {str(ex)}")

# Main Premium Footer
st.markdown("""
<div style="border-top: 1px solid var(--border-color); margin-top: 50px; padding-top: 30px; padding-bottom: 20px; text-align: center; color: var(--text-muted); font-size: 0.9rem;">
    <div style="display: flex; justify-content: center; align-items: center; gap: 8px; margin-bottom: 8px;">
        <span style="font-size: 1.5rem;">🩺</span>
        <span style="font-size: 1.2rem; font-weight: 700; background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">MedSimplify</span>
    </div>
    <p style="margin: 0 0 15px 0; font-size: 0.85rem;">AI-Powered Healthcare Assistant</p>
    <div style="max-width: 600px; margin: 0 auto; background: var(--meta-card-bg); padding: 15px 25px; border-radius: 12px; border: 1px solid var(--border-color);">
        <p style="text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em; font-weight: 600; margin-bottom: 8px; color: var(--text-primary);">Developed By</p>
        <div style="display: flex; justify-content: space-around; flex-wrap: wrap; gap: 15px;">
            <div>
                <b style="color: var(--accent-primary);">Jayashri V. Hiremath</b><br>
                <span style="font-size: 0.8rem;">B.Tech CSE</span>
            </div>
            <div>
                <b style="color: var(--accent-primary);">Tanishka Desai</b><br>
                <span style="font-size: 0.8rem;">B.Tech CSI</span>
            </div>
        </div>
        <p style="font-size: 0.8rem; margin-top: 10px; margin-bottom: 0;">Presidency University</p>
    </div>
</div>
""", unsafe_allow_html=True)
