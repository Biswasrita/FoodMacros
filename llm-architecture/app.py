import base64
import os
import sys
import textwrap
import cv2
import numpy as np
from PIL import Image, ImageDraw
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Append current directory for modular imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

# Modular imports with fallbacks
from models.detector import FoodDetector
from models.coach_agent import FudAICoach
from utils.deduplication import filter_redundant_crops
from utils.nutrition_tracker import initialize_nutrition_state, log_meal, get_daily_summary
from utils.workout_tracker import initialize_workout_state, log_exercise_set
from utils.fasting_water import initialize_fasting_state, render_fasting_widget, log_water
from utils.calculators import calculate_bmr_tdee

try:
    from models.llm_classifier_overcooked import LLMFoodClassifier
except ImportError:
    from models.llm_classifier import LLMFoodClassifier

# Set Page Config - Dark Layout & Hidden Sidebar
st.set_page_config(
    page_title="DIET ENGINE — Dark Vision & Performance Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize Session States
initialize_nutrition_state()
initialize_workout_state()
initialize_fasting_state()

# Bulletproof HTML Renderer (strips line indentation so Markdown never breaks HTML tags)
def render_html(html_str: str):
    clean_html = "\n".join(line.strip() for line in html_str.splitlines())
    st.markdown(clean_html, unsafe_allow_html=True)

# Helper function to convert local video to base64
def get_base64_video(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return None

# Locate background video file with robust multi-path fallbacks
possible_video_paths = [
    os.path.join(CURRENT_DIR, "assets", "motionpage-silk-ribbons.mp4"),
    os.path.join(CURRENT_DIR, "motionpage-silk-ribbons.mp4"),
    os.path.join(os.getcwd(), "assets", "motionpage-silk-ribbons.mp4"),
    os.path.join(os.getcwd(), "motionpage-silk-ribbons.mp4"),
]
video_b64 = None
for v_path in possible_video_paths:
    if os.path.exists(v_path):
        video_b64 = get_base64_video(v_path)
        if video_b64:
            break

# Dark Theme CSS + Glass Card Layout
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    :root {
        --bg-dark: #060911;
        --card-bg: rgba(15, 23, 42, 0.88);
        --card-border: rgba(56, 189, 248, 0.25);
        --primary-cyan: #00f2fe;
        --accent-blue: #38bdf8;
        --accent-pink: #ec4899;
        --accent-purple: #a855f7;
        --text-bright: #f8fafc;
        --text-muted: #94a3b8;
    }

    /* 1. COMPLETELY HIDE STREAMLIT SIDEBAR & TOP BAR */
    [data-testid="stSidebar"], section[data-testid="stSidebar"] {
        display: none !important;
        width: 0px !important;
    }
    
    .stApp > header {
        background: transparent !important;
    }

    /* 2. DARK STREAMLIT BASE */
    .stApp {
        background-color: var(--bg-dark) !important;
        color: var(--text-bright) !important;
        font-family: 'Plus Jakarta Sans', sans-serif;
        overflow-x: hidden;
    }

    /* Text Visibility Overrides */
    h1, h2, h3, h4, h5, h6, p, label, span, div {
        color: var(--text-bright) !important;
    }

    /* 3. WAVY FALLBACK SVG ANIMATION (Used if video file is missing) */
    .wavy-bg-viewport {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        z-index: 0;
        pointer-events: none;
        overflow: hidden;
        background: radial-gradient(circle at 50% 20%, #111827 0%, #060911 100%);
    }

    .wavy-svg {
        position: absolute;
        width: 120%;
        height: 120%;
        top: -10%;
        left: -10%;
        opacity: 0.65;
    }

    .animated-wave {
        fill: none;
        stroke-linecap: round;
        animation: waveFlow 14s infinite ease-in-out alternate;
    }

    .wave-1 { stroke: url(#pinkGradient); stroke-width: 4; animation-duration: 16s; }
    .wave-2 { stroke: url(#purpleGradient); stroke-width: 5; animation-duration: 20s; animation-delay: -4s; }
    .wave-3 { stroke: url(#cyanGradient); stroke-width: 3; animation-duration: 13s; animation-delay: -2s; }

    @keyframes waveFlow {
        0% { transform: translateY(0px) scaleY(1) rotate(0deg); }
        50% { transform: translateY(-45px) scaleY(1.2) rotate(1deg); }
        100% { transform: translateY(35px) scaleY(0.9) rotate(-1deg); }
    }

    /* 4. DARK GLASS CARDS BEHIND ALL TEXT & SECTIONS */
    .dark-card {
        position: relative;
        z-index: 2;
        background: var(--card-bg);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1.5px solid var(--card-border);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.6), 0 0 20px rgba(0, 242, 254, 0.06);
        margin-bottom: 1.4rem;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }

    .dark-card:hover {
        border-color: rgba(0, 242, 254, 0.55);
        box-shadow: 0 20px 48px rgba(0, 0, 0, 0.75), 0 0 30px rgba(0, 242, 254, 0.15);
    }

    .dark-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1.5px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 0.85rem;
        margin-bottom: 1.2rem;
    }

    .dark-card-title {
        font-weight: 800;
        font-size: 1.25rem;
        letter-spacing: 0.5px;
        color: #38bdf8 !important;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }

    .pulse-badge {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 700;
        color: #00f2fe !important;
        background: rgba(0, 242, 254, 0.12);
        border: 1px solid rgba(0, 242, 254, 0.3);
        padding: 0.35rem 0.9rem;
        border-radius: 9999px;
    }

    .pulse-dot {
        width: 8px;
        height: 8px;
        background: #00f2fe;
        border-radius: 50%;
        box-shadow: 0 0 10px #00f2fe;
        animation: pulse 1.8s infinite;
    }

    @keyframes pulse {
        0% { transform: scale(0.9); opacity: 0.7; }
        50% { transform: scale(1.3); opacity: 1; }
        100% { transform: scale(0.9); opacity: 0.7; }
    }

    /* 5. HERO CONTAINER & SECTION HEADERS */
    .hero-container {
        position: relative;
        z-index: 2;
        text-align: center;
        padding: 1.5rem 1rem 1.8rem 1rem;
        margin-bottom: 1.2rem;
    }

    .tech-tag {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 2px;
        color: #00f2fe !important;
        background: rgba(0, 242, 254, 0.1);
        border: 1.5px solid rgba(0, 242, 254, 0.35);
        padding: 0.4rem 1.2rem;
        border-radius: 9999px;
        margin-bottom: 0.8rem;
        text-transform: uppercase;
        box-shadow: 0 0 15px rgba(0, 242, 254, 0.2);
    }

    .diet-title {
        font-size: 3.4rem;
        font-weight: 900;
        letter-spacing: -1.5px;
        background: linear-gradient(135deg, #ffffff 0%, #38bdf8 40%, #00f2fe 75%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.1;
    }

    .diet-subtitle {
        font-size: 1.15rem;
        font-weight: 600;
        color: #94a3b8 !important;
        margin-top: 0.4rem;
    }

    .section-title-card {
        position: relative;
        z-index: 2;
        background: rgba(15, 23, 42, 0.9);
        border: 1.5px solid rgba(56, 189, 248, 0.3);
        border-left: 5px solid #00f2fe;
        border-radius: 14px;
        padding: 0.9rem 1.4rem;
        margin: 2rem 0 1.2rem 0;
        font-size: 1.4rem;
        font-weight: 800;
        color: #f8fafc !important;
        display: flex;
        align-items: center;
        gap: 0.7rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.4);
    }

    /* 6. HERO CALORIE DISPLAY & PILL STATS */
    .cal-hero-dark {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.9) 100%);
        border: 1.5px solid rgba(56, 189, 248, 0.3);
        border-radius: 18px;
        padding: 1.3rem;
        text-align: center;
        margin: 1rem 0 1.3rem 0;
        box-shadow: inset 0 0 20px rgba(0, 242, 254, 0.05);
    }

    .cal-title-dark {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.6px;
        color: #94a3b8 !important;
        font-weight: 700;
    }

    .cal-val-dark {
        font-size: 3.2rem;
        font-weight: 900;
        color: #ffffff !important;
        line-height: 1.1;
        margin: 0.2rem 0;
    }

    .cal-val-dark span {
        font-size: 1.3rem;
        font-weight: 800;
        color: #00f2fe !important;
    }

    .pill-grid-dark {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.85rem;
        margin: 1.1rem 0;
    }

    .stat-pill-dark {
        background: rgba(30, 41, 59, 0.8);
        border: 1.5px solid rgba(148, 163, 184, 0.2);
        padding: 0.85rem 1rem;
        border-radius: 16px;
        text-align: center;
    }

    .stat-pill-dark .pill-num {
        font-size: 1.35rem;
        font-weight: 800;
        color: #f8fafc !important;
    }

    .stat-pill-dark .pill-txt {
        font-size: 0.78rem;
        color: #94a3b8 !important;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.8px;
        margin-top: 0.25rem;
    }

    /* 7. MACRONUTRIENT BARS */
    .macro-section-title-dark {
        font-size: 1.05rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1.4px;
        color: #38bdf8 !important;
        margin-bottom: 1rem;
        border-left: 3px solid #00f2fe;
        padding-left: 0.6rem;
    }

    .macro-row-dark { margin-bottom: 1.1rem; }
    .macro-label-row-dark {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.95rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
        color: #e2e8f0 !important;
    }

    .macro-track-dark {
        width: 100%;
        height: 12px;
        background: rgba(30, 41, 59, 0.9);
        border-radius: 9999px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .macro-fill-dark { height: 100%; border-radius: 9999px; }
    .fill-pro { background: linear-gradient(90deg, #00f2fe, #0284c7); }
    .fill-carb { background: linear-gradient(90deg, #38bdf8, #0ea5e9); }
    .fill-fat { background: linear-gradient(90deg, #10b981, #059669); }

    /* 8. CONDITION BADGES */
    .condition-badge-dark {
        padding: 1rem 1.2rem;
        border-radius: 16px;
        font-size: 0.95rem;
        margin-top: 1.2rem;
        display: flex;
        align-items: flex-start;
        gap: 0.8rem;
        line-height: 1.5;
    }

    .cond-normal { background: rgba(16, 185, 129, 0.15); border: 1.5px solid #10b981; color: #a7f3d0 !important; }
    .cond-overcooked { background: rgba(245, 158, 11, 0.15); border: 1.5px solid #f59e0b; color: #fde68a !important; }
    .cond-spoilage { background: rgba(244, 63, 94, 0.15); border: 1.5px solid #f43f5e; color: #fecdd3 !important; }
    .cond-uncertain { background: rgba(148, 163, 184, 0.15); border: 1.5px solid #94a3b8; color: #e2e8f0 !important; }

    /* 9. STREAMLIT FORM CONTROLS DARK OVERRIDES */
    [data-testid="stFileUploader"] {
        background: rgba(15, 23, 42, 0.85) !important;
        border: 2px dashed rgba(56, 189, 248, 0.4) !important;
        border-radius: 20px !important;
        padding: 1.4rem !important;
        color: #f8fafc !important;
    }

    div[data-baseweb="input"], div[data-baseweb="select"], div[role="combobox"] {
        background-color: #0f172a !important;
        border-color: rgba(56, 189, 248, 0.3) !important;
        color: #f8fafc !important;
        border-radius: 12px !important;
    }

    input, textarea {
        color: #f8fafc !important;
        background-color: #0f172a !important;
    }

    .stNumberInput input, .stTextInput input, .stTextArea textarea {
        background-color: #0f172a !important;
        color: #f8fafc !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
        border-radius: 10px !important;
    }

    .stButton button {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        border-radius: 12px !important;
        padding: 0.6rem 1.4rem !important;
        border: 1px solid rgba(0, 242, 254, 0.3) !important;
        box-shadow: 0 4px 16px rgba(2, 132, 199, 0.4) !important;
        transition: all 0.25s ease !important;
    }

    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(0, 242, 254, 0.5) !important;
        border-color: #00f2fe !important;
    }

    [data-testid="stRadio"] label {
        color: #f8fafc !important;
    }

    .streamlit-expanderHeader {
        background-color: rgba(15, 23, 42, 0.9) !important;
        color: #38bdf8 !important;
        border-radius: 12px !important;
        border: 1px solid rgba(56, 189, 248, 0.2) !important;
    }

    [data-testid="stMetricValue"] {
        color: #00f2fe !important;
        font-weight: 800 !important;
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 700 !important;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Render Background: Embedded Base64 Video OR Fallback Animated Wavy SVG
if video_b64:
    render_html(f"""
    <video autoplay loop muted playsinline style="
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        object-fit: cover;
        z-index: 0;
        opacity: 0.45;
        pointer-events: none;
    ">
        <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
    </video>
    """)
else:
    render_html("""
    <div class="wavy-bg-viewport">
        <svg class="wavy-svg" viewBox="0 0 1440 900" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">
            <defs>
                <linearGradient id="pinkGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stop-color="#ec4899" stop-opacity="0.8"/>
                    <stop offset="50%" stop-color="#f43f5e" stop-opacity="0.9"/>
                    <stop offset="100%" stop-color="#fb7185" stop-opacity="0.7"/>
                </linearGradient>
                <linearGradient id="purpleGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stop-color="#8b5cf6" stop-opacity="0.7"/>
                    <stop offset="50%" stop-color="#d946ef" stop-opacity="0.9"/>
                    <stop offset="100%" stop-color="#a855f7" stop-opacity="0.8"/>
                </linearGradient>
                <linearGradient id="cyanGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stop-color="#06b6d4" stop-opacity="0.6"/>
                    <stop offset="50%" stop-color="#00f2fe" stop-opacity="0.85"/>
                    <stop offset="100%" stop-color="#38bdf8" stop-opacity="0.7"/>
                </linearGradient>
            </defs>
            <path class="animated-wave wave-1" d="M -100 150 Q 300 280, 700 160 T 1500 200" />
            <path class="animated-wave wave-2" d="M -100 300 Q 400 180, 800 340 T 1500 290" />
            <path class="animated-wave wave-3" d="M -100 450 Q 350 560, 750 420 T 1500 480" />
        </svg>
    </div>
    """)

# Draw High-Tech Overlay Brackets on Image
def draw_high_tech_overlay(image_pil, boxes, item_labels=None):
    img = image_pil.copy()
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size

    box_color = (0, 242, 254, 255)
    bracket_color = (0, 242, 254, 255)
    fill_color = (0, 242, 254, 35)
    badge_bg = (15, 23, 42, 240)
    badge_text_color = (255, 255, 255, 255)

    bracket_len = max(16, int(min(w, h) * 0.04))
    line_thickness = max(2, int(min(w, h) * 0.0035))

    for idx, b in enumerate(boxes):
        x1, y1, x2, y2 = b['box']
        draw.rectangle([x1, y1, x2, y2], fill=fill_color, outline=box_color, width=line_thickness)

        draw.line([(x1, y1), (x1 + bracket_len, y1)], fill=bracket_color, width=line_thickness * 2)
        draw.line([(x1, y1), (x1, y1 + bracket_len)], fill=bracket_color, width=line_thickness * 2)
        draw.line([(x2, y1), (x2 - bracket_len, y1)], fill=bracket_color, width=line_thickness * 2)
        draw.line([(x2, y1), (x2, y1 + bracket_len)], fill=bracket_color, width=line_thickness * 2)
        draw.line([(x1, y2), (x1 + bracket_len, y2)], fill=bracket_color, width=line_thickness * 2)
        draw.line([(x1, y2), (x1, y2 - bracket_len)], fill=bracket_color, width=line_thickness * 2)
        draw.line([(x2, y2), (x2 - bracket_len, y2)], fill=bracket_color, width=line_thickness * 2)
        draw.line([(x2, y2), (x2, y2 - bracket_len)], fill=bracket_color, width=line_thickness * 2)

        name = item_labels[idx] if (item_labels and idx < len(item_labels)) else f"Item {idx + 1}"
        badge_text = f" [{idx + 1:02d}] {name.title()} "

        badge_w = len(badge_text) * 9 + 16
        badge_h = 26
        by1 = max(0, y1 - badge_h - 4)
        by2 = by1 + badge_h
        bx2 = min(w, x1 + badge_w)
        draw.rounded_rectangle([x1, by1, bx2, by2], radius=6, fill=badge_bg, outline=box_color, width=1)
        draw.text((x1 + 8, by1 + 5), badge_text, fill=badge_text_color)

    return img

def normalize_item_dict(item):
    return {
        "label": str(item.get("label", "unknown")).strip().lower(),
        "portion_unit": str(item.get("portion_unit", "1 portion")).strip(),
        "confidence": float(item.get("confidence", 0.9)),
        "calories": float(item.get("calories", 0.0)),
        "protein": float(item.get("protein", item.get("protein_g", 0.0))),
        "fat": float(item.get("fat", item.get("fat_g", 0.0))),
        "carbs": float(item.get("carbs", item.get("carbs_g", 0.0))),
        "fiber": float(item.get("fiber", item.get("fiber_g", 0.0))),
        "estimated_weight_g": float(item.get("estimated_weight_g", 100.0)),
        "visual_condition": str(item.get("visual_condition", "normal_appearance")),
        "condition_severity": str(item.get("condition_severity", "none")),
        "condition_evidence": str(item.get("condition_evidence", "Normal visual appearance."))
    }

# Automatic API Key Loading
api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("api_key", "")

if not api_key:
    st.error("⚠️ Gemini API Key missing! Please set `GEMINI_API_KEY` in your `.env` file or environment variables.")
    st.stop()

@st.cache_resource
def load_models(key):
    detector = FoodDetector()
    classifier = LLMFoodClassifier(api_key=key)
    coach = FudAICoach(api_key=key)
    return detector, classifier, coach

try:
    detector, classifier, coach = load_models(api_key)
except Exception as e:
    st.error(f"Failed to load AI models: {e}")
    st.stop()

# HERO HEADER
render_html("""
<div class="hero-container">
    <div class="tech-tag">RESEARCH PROJECT &bull; YOLOV8 &bull; GEMINI VISION &bull; MULTI-DISH AI</div>
    <h1 class="diet-title">DIET ENGINE</h1>
    <p class="diet-subtitle">All-in-One AI Nutrition, Fasting, Fitness & Performance Platform</p>
</div>
""")

# ==============================================================================
# SECTION 1: DAILY MACRO DASHBOARD & FOOD SCANNER
# ==============================================================================
render_html('<div class="section-title-card">⚡ Today\'s Nutrition & AI Vision Logger</div>')

summary = get_daily_summary()

# Top Metrics Row inside a Dark Card
st.markdown('<div class="dark-card">', unsafe_allow_html=True)
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Calories Target", f"{summary['calories']:.0f} / {st.session_state.get('target_cal', 2000)} kcal")
m2.metric("Protein Intake", f"{summary['protein']:.1f} g")
m3.metric("Carbohydrates", f"{summary['carbs']:.1f} g")
m4.metric("Healthy Fats", f"{summary['fat']:.1f} g")
m5.metric("Water Hydration", f"{st.session_state.get('water_ml', 0)} / {st.session_state.get('target_water', 2500)} ml")
st.markdown('</div>', unsafe_allow_html=True)

# Input Method Selector inside a Dark Card
st.markdown('<div class="dark-card">', unsafe_allow_html=True)
log_method = st.radio("Choose Input Method", ["📸 Photo / Camera Scan", "📝 Natural Text / Speech", "🔍 Barcode Text"], horizontal=True)

if log_method == "📸 Photo / Camera Scan":
    uploaded_file = st.file_uploader("Upload or Snap a Meal Photo", type=["jpg", "jpeg", "png", "webp"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)

        with st.spinner("⚡ Running YOLOv8 Detection & Gemini Vision AI..."):
            raw_boxes, raw_crops = detector.detect(image)
            boxes, crops = filter_redundant_crops(raw_boxes, raw_crops)

            items_data = []
            item_labels = []

            if crops:
                if hasattr(classifier, "predict_all"):
                    raw_items = classifier.predict_all(crops)
                    items_data = [normalize_item_dict(it) for it in raw_items]
                else:
                    for crop in crops:
                        raw_it = classifier.predict(crop)
                        items_data.append(normalize_item_dict(raw_it))
                
                item_labels = [it["label"] for it in items_data]

        col1, col2 = st.columns([1.1, 1], gap="large")

        with col1:
            st.markdown('<div class="dark-card">', unsafe_allow_html=True)
            annotated_img = draw_high_tech_overlay(image, boxes, item_labels)
            st.image(annotated_img, caption="YOLOv8 Dynamic Target Localization", use_container_width=True)

            num_detected = len(boxes)
            render_html(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 0.8rem; padding: 0.75rem 1rem; background: rgba(30,41,59,0.8); border: 1.5px solid rgba(56,189,248,0.3); border-radius: 12px; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;">
                <div><span>TARGETS DETECTED:</span> <strong style="color: #00f2fe; font-size: 1rem;">{num_detected}</strong></div>
                <div><span>STATUS:</span> <strong style="color: #10b981; font-size: 0.9rem;">OPTIMAL SCAN</strong></div>
            </div>
            """)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            total_cal = sum(it["calories"] for it in items_data)
            total_pro = sum(it["protein"] for it in items_data)
            total_fat = sum(it["fat"] for it in items_data)
            total_carb = sum(it["carbs"] for it in items_data)
            total_fiber = sum(it["fiber"] for it in items_data)
            total_weight = sum(it["estimated_weight_g"] for it in items_data)

            conditions = [it["visual_condition"] for it in items_data]
            if "possible_spoilage_signs" in conditions:
                dominant_cond = "possible_spoilage_signs"
            elif "overcooked_or_burnt" in conditions:
                dominant_cond = "overcooked_or_burnt"
            elif "normal_appearance" in conditions:
                dominant_cond = "normal_appearance"
            else:
                dominant_cond = "uncertain"

            composite_meal_name = " + ".join([it["label"].title() for it in items_data[:3]]) if items_data else "Meal Scan"

            # Dark HUD Card
            render_html(f"""
            <div class="dark-card">
                <div class="dark-card-header">
                    <div class="dark-card-title"><span>⚡</span> DIET ENGINE</div>
                    <div class="pulse-badge"><span class="pulse-dot"></span> SCAN COMPLETE</div>
                </div>

                <div style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1.2px; color: #94a3b8; font-weight: 700;">Meal Composition</div>
                <div style="font-size: 1.45rem; font-weight: 800; color: #ffffff; margin-bottom: 0.6rem;">{composite_meal_name}</div>

                <div class="cal-hero-dark">
                    <div class="cal-title-dark">Total Energy Value</div>
                    <div class="cal-val-dark">{total_cal:.0f} <span>kcal</span></div>
                    <div style="font-size: 0.88rem; color: #94a3b8; font-weight: 600;">Computed across {num_detected} isolated portion{'s' if num_detected != 1 else ''}</div>
                </div>

                <div class="pill-grid-dark">
                    <div class="stat-pill-dark">
                        <div class="pill-num">⚖️ {total_weight:.0f}g</div>
                        <div class="pill-txt">Est. Food Weight</div>
                    </div>
                    <div class="stat-pill-dark">
                        <div class="pill-num">🌾 {total_fiber:.1f}g</div>
                        <div class="pill-txt">Dietary Fiber</div>
                    </div>
                </div>

                <div style="margin: 1.4rem 0 1rem 0;">
                    <div class="macro-section-title-dark">Macronutrient Profile</div>
                    <div class="macro-row-dark">
                        <div class="macro-label-row-dark"><span>🍗 Protein</span><span style="color: #00f2fe;">{total_pro:.1f}g</span></div>
                        <div class="macro-track-dark"><div class="macro-fill-dark fill-pro" style="width: {min(100, int((total_pro / 120.0) * 100))}%;"></div></div>
                    </div>
                    <div class="macro-row-dark">
                        <div class="macro-label-row-dark"><span>🍞 Carbohydrates</span><span style="color: #38bdf8;">{total_carb:.1f}g</span></div>
                        <div class="macro-track-dark"><div class="macro-fill-dark fill-carb" style="width: {min(100, int((total_carb / 250.0) * 100))}%;"></div></div>
                    </div>
                    <div class="macro-row-dark">
                        <div class="macro-label-row-dark"><span>🥑 Healthy Fats</span><span style="color: #10b981;">{total_fat:.1f}g</span></div>
                        <div class="macro-track-dark"><div class="macro-fill-dark fill-fat" style="width: {min(100, int((total_fat / 80.0) * 100))}%;"></div></div>
                    </div>
                </div>
            </div>
            """)

            if dominant_cond == "overcooked_or_burnt":
                render_html('<div class="condition-badge-dark cond-overcooked">🔥 <div><strong>Overcooked/Burnt Surface Detected</strong><br>Visible charring or heavy surface darkening identified.</div></div>')
            elif dominant_cond == "possible_spoilage_signs":
                render_html('<div class="condition-badge-dark cond-spoilage">⚠️ <div><strong>Visual Spoilage Signs Detected</strong><br>Abnormal discoloration or surface deterioration identified.</div></div>')
            elif dominant_cond == "normal_appearance":
                render_html('<div class="condition-badge-dark cond-normal">✨ <div><strong>Optimal Visual Quality</strong><br>Fresh appearance with no obvious signs of charring or spoilage.</div></div>')
            else:
                render_html('<div class="condition-badge-dark cond-uncertain">ℹ️ <div><strong>Condition Assessment Uncertain</strong><br>Visual indicators are inconclusive.</div></div>')

            st.subheader("Review & Save Items to Daily Log")
            for i, item in enumerate(items_data):
                with st.expander(f"Item {i+1}: {item['portion_unit']} of {item['label'].title()} ({item['calories']:.0f} kcal)", expanded=(i == 0)):
                    c1, c2 = st.columns(2)
                    item['label'] = c1.text_input(f"Dish Name #{i+1}", value=item['label'])
                    item['calories'] = c2.number_input(f"Calories (kcal) #{i+1}", value=float(item['calories']))

                    c3, c4, c5 = st.columns(3)
                    item['protein'] = c3.number_input(f"Protein (g) #{i+1}", value=float(item['protein']))
                    item['carbs'] = c4.number_input(f"Carbs (g) #{i+1}", value=float(item['carbs']))
                    item['fat'] = c5.number_input(f"Fat (g) #{i+1}", value=float(item['fat']))

                    st.caption(f"**Visual Condition:** {item['visual_condition'].replace('_', ' ').title()} | **Evidence:** {item['condition_evidence']}")

                    if st.button(f"📥 Save Item #{i+1} to Daily Tracker", key=f"save_scan_{i}"):
                        log_meal(item)
                        st.success(f"Logged {item['label'].title()} ({item['calories']} kcal) to daily summary!")
                        st.rerun()

elif log_method == "📝 Natural Text / Speech":
    text_prompt = st.text_area("Describe your meal (e.g., '2 scrambled eggs with a slice of sourdough toast and black coffee')")
    if st.button("Parse & Log Text"):
        with st.spinner("AI parsing meal description..."):
            raw_item = classifier.parse_text_description(text_prompt)
            item = normalize_item_dict(raw_item if isinstance(raw_item, dict) else raw_item[0])
            log_meal(item)
            st.success(f"Logged {item['portion_unit']} of {item['label']} ({item['calories']} kcal)!")
            st.rerun()

elif log_method == "🔍 Barcode Text":
    barcode_input = st.text_input("Enter Barcode Number or Packaged Product Name")
    if st.button("Lookup & Log"):
        with st.spinner("Querying nutrition registry..."):
            raw_item = classifier.parse_text_description(f"Barcode or packaged product lookup for: {barcode_input}")
            item = normalize_item_dict(raw_item if isinstance(raw_item, dict) else raw_item[0])
            log_meal(item)
            st.success(f"Logged {item['label']} ({item['calories']} kcal)!")
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: FASTING & WATER HYDRATION MATRIX
# ==============================================================================
render_html('<div class="section-title-card">⏱️ Fasting & Water Hydration Matrix</div>')
f_col1, f_col2 = st.columns([1.2, 1], gap="large")

with f_col1:
    st.markdown('<div class="dark-card">', unsafe_allow_html=True)
    render_html("""
    <div class="dark-card-header">
        <div class="dark-card-title"><span>⏱️</span> INTERMITTENT FASTING</div>
        <div class="pulse-badge"><span class="pulse-dot"></span> LIVE TIMER</div>
    </div>
    """)
    render_fasting_widget()
    st.markdown('</div>', unsafe_allow_html=True)

with f_col2:
    st.markdown('<div class="dark-card">', unsafe_allow_html=True)
    render_html("""
    <div class="dark-card-header">
        <div class="dark-card-title"><span>💧</span> DAILY WATER TRACKER</div>
        <div class="pulse-badge"><span class="pulse-dot"></span> HYDRATION</div>
    </div>
    """)
    w1, w2, w3 = st.columns(3)
    if w1.button("+ 250 ml"):
        log_water(250)
        st.rerun()
    if w2.button("+ 500 ml"):
        log_water(500)
        st.rerun()
    if w3.button("Reset"):
        st.session_state['water_ml'] = 0
        st.rerun()

    render_html(f"""
    <div class="cal-hero-dark" style="margin-top: 1rem;">
        <div class="cal-title-dark">Logged Hydration</div>
        <div class="cal-val-dark">{st.session_state.get('water_ml', 0)} <span>ml</span></div>
        <div style="font-size: 0.85rem; color: #94a3b8;">Daily Target: {st.session_state.get('target_water', 2500)} ml</div>
    </div>
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# SECTION 3: WORKOUT & ATHLETIC PERFORMANCE
# ==============================================================================
render_html('<div class="section-title-card">🏋️ Athletic Performance & Set Tracker</div>')
w_col1, w_col2 = st.columns([1, 1.3], gap="large")

with w_col1:
    st.markdown('<div class="dark-card">', unsafe_allow_html=True)
    render_html("""
    <div class="dark-card-header">
        <div class="dark-card-title"><span>🏋️</span> LOG EXERCISE SET</div>
        <div class="pulse-badge"><span class="pulse-dot"></span> ACTIVE SESSION</div>
    </div>
    """)
    exercise = st.text_input("Exercise Name", placeholder="e.g., Bench Press, Squat, Pull-ups")
    weight = st.number_input("Weight (kg / lbs)", min_value=0.0, value=60.0)
    reps = st.number_input("Reps", min_value=1, value=10)
    rpe = st.slider("RPE (Rate of Perceived Exertion)", 1, 10, 8)

    if st.button("➕ Add Set to Session"):
        log_exercise_set(exercise, weight, reps, rpe)
        st.success(f"Logged {reps} reps of {exercise}!")
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with w_col2:
    st.markdown('<div class="dark-card">', unsafe_allow_html=True)
    render_html("""
    <div class="dark-card-header">
        <div class="dark-card-title"><span>📜</span> TODAY\'S WORKOUT LOG</div>
        <div class="pulse-badge"><span class="pulse-dot"></span> RECORDED SETS</div>
    </div>
    """)
    if st.session_state.get('workout_logs'):
        for i, w in enumerate(st.session_state['workout_logs']):
            render_html(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.75rem 1rem; background: rgba(30,41,59,0.8); border: 1.5px solid rgba(56,189,248,0.3); border-radius: 12px; margin-bottom: 0.6rem;">
                <div style="font-weight: 800; color: #ffffff; font-size: 1.05rem;">{i+1}. {w['exercise'].title()}</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; color: #00f2fe; font-weight: 700;">
                    {w['reps']} reps @ {w['weight']}kg <span style="color: #94a3b8; font-size: 0.8rem;">(RPE {w['rpe']})</span>
                </div>
            </div>
            """)
    else:
        st.info("No exercises logged today yet. Use the form on the left to add sets.")
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# SECTION 4: BMR, TDEE & TARGET CALCULATOR
# ==============================================================================
render_html('<div class="section-title-card">🧮 Metabolic Rate & TDEE Target Engine</div>')
c_col1, c_col2 = st.columns([1, 1.2], gap="large")

with c_col1:
    st.markdown('<div class="dark-card">', unsafe_allow_html=True)
    render_html("""
    <div class="dark-card-header">
        <div class="dark-card-title"><span>⚙️</span> METABOLIC INPUTS</div>
        <div class="pulse-badge"><span class="pulse-dot"></span> CALCULATOR</div>
    </div>
    """)
    weight_kg = st.number_input("Weight (kg)", value=70.0)
    height_cm = st.number_input("Height (cm)", value=175.0)
    age = st.number_input("Age", value=25)
    gender = st.selectbox("Gender", ["Male", "Female"])
    activity = st.selectbox("Activity Level", ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"])

    if st.button("🧮 Compute Metabolic Targets"):
        bmr, tdee = calculate_bmr_tdee(weight_kg, height_cm, age, gender, activity)
        st.session_state['target_cal'] = tdee
        st.success(f"**BMR:** {bmr:.0f} kcal | **TDEE (Daily Target):** {tdee:.0f} kcal")
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with c_col2:
    target_cal_val = st.session_state.get('target_cal', 2000)
    st.markdown('<div class="dark-card">', unsafe_allow_html=True)
    render_html(f"""
    <div class="dark-card-header">
        <div class="dark-card-title"><span>🎯</span> TARGET ENERGY SUMMARY</div>
        <div class="pulse-badge"><span class="pulse-dot"></span> ACTIVE GOAL</div>
    </div>
    
    <div class="cal-hero-dark">
        <div class="cal-title-dark">Computed Daily Maintenance Target (TDEE)</div>
        <div class="cal-val-dark">{target_cal_val:.0f} <span>kcal/day</span></div>
        <div style="font-size: 0.88rem; color: #94a3b8; margin-top: 0.3rem;">Synced automatically across your daily dashboard bars</div>
    </div>

    <div class="pill-grid-dark">
        <div class="stat-pill-dark">
            <div class="pill-num">💪 {(target_cal_val * 0.25 / 4):.0f}g</div>
            <div class="pill-txt">Rec. Protein</div>
        </div>
        <div class="stat-pill-dark">
            <div class="pill-num">🍞 {(target_cal_val * 0.50 / 4):.0f}g</div>
            <div class="pill-txt">Rec. Carbs</div>
        </div>
    </div>
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# SECTION 5: AI HEALTH & WELLNESS COACH TERMINAL
# ==============================================================================
render_html('<div class="section-title-card">💬 Fud AI Intelligent Health Coach</div>')

st.markdown('<div class="dark-card">', unsafe_allow_html=True)
render_html("""
<div class="dark-card-header">
    <div class="dark-card-title"><span>🤖</span> AI COACH TERMINAL</div>
    <div class="pulse-badge"><span class="pulse-dot"></span> CONTEXT-AWARE</div>
</div>
""")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask about your diet, workouts, or fasting window..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("AI Coach is analyzing live session context..."):
            context = {
                "daily_summary": get_daily_summary(),
                "fasting": st.session_state.get("fasting_state"),
                "workouts": st.session_state.get("workout_logs", [])
            }
            response = coach.chat(user_input, context, st.session_state.messages)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown('</div>', unsafe_allow_html=True)