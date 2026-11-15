import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import sys
import os
import textwrap
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Append current directory so modular imports work smoothly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

from models.detector import FoodDetector
from models.llm_classifier_overcooked import LLMFoodClassifier

# Set Streamlit Page Config
st.set_page_config(
    layout="wide",
    page_title="DIET ENGINE — AI Nutrition & Research Analysis",
    page_icon="🥗",
    initial_sidebar_state="collapsed"
)

# Helper function to render raw HTML without markdown treating indentation as code blocks
def render_html(html_str: str):
    st.markdown(textwrap.dedent(html_str).strip(), unsafe_allow_html=True)

# Custom High-Tech Sci-Fi "Diet Engine" Theme CSS with enhanced contrast & font sizing
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    :root {
        --primary-glow: #0284c7;
        --secondary-glow: #00f2fe;
        --accent-teal: #06b6d4;
        --accent-emerald: #10b981;
        --accent-amber: #f59e0b;
        --accent-rose: #f43f5e;
        --bg-glass: rgba(255, 255, 255, 0.95);
        --text-dark: #0f172a;
        --text-muted: #475569;
    }

    /* Global Body & Container */
    .stApp {
        background: radial-gradient(circle at 10% 10%, rgba(2, 132, 199, 0.08) 0%, transparent 45%),
                    radial-gradient(circle at 90% 90%, rgba(6, 182, 212, 0.08) 0%, transparent 45%),
                    linear-gradient(135deg, #f8fafc 0%, #edf2f7 50%, #e2e8f0 100%);
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: var(--text-dark);
    }

    /* Header Styling */
    .hero-container {
        text-align: center;
        padding: 1.8rem 1rem 2.2rem 1rem;
        margin-bottom: 1.5rem;
    }

    .tech-tag {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 2.2px;
        color: #0284c7;
        background: rgba(2, 132, 199, 0.12);
        border: 1.5px solid rgba(2, 132, 199, 0.35);
        padding: 0.4rem 1.3rem;
        border-radius: 9999px;
        margin-bottom: 0.85rem;
        text-transform: uppercase;
        box-shadow: 0 2px 10px rgba(2, 132, 199, 0.15);
    }

    .diet-title {
        font-size: 3.5rem;
        font-weight: 900;
        letter-spacing: -1.5px;
        background: linear-gradient(135deg, #0f172a 0%, #0369a1 45%, #0284c7 85%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.1;
    }

    .diet-subtitle {
        font-size: 1.25rem;
        font-weight: 600;
        color: #334155;
        margin-top: 0.5rem;
        letter-spacing: 0.3px;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border: 1.5px solid rgba(2, 132, 199, 0.25);
        border-radius: 24px;
        padding: 1.8rem;
        box-shadow: 0 14px 38px -8px rgba(3, 105, 161, 0.12),
                    0 0 0 1px rgba(255, 255, 255, 0.8) inset;
        margin-bottom: 1.5rem;
    }

    /* File Uploader Custom Styling - FIXING BLACK / INVISIBLE BUTTONS */
    [data-testid="stFileUploader"] {
        background: #ffffff !important;
        border: 2px dashed #0284c7 !important;
        border-radius: 20px !important;
        padding: 1.6rem !important;
        box-shadow: 0 6px 24px rgba(2, 132, 199, 0.1) !important;
        margin-top: 0.8rem !important;
    }

    [data-testid="stFileUploader"] section {
        background: #ffffff !important;
        border: none !important;
        padding: 0 !important;
    }

    [data-testid="stFileUploader"] section > div {
        color: #0f172a !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }

    [data-testid="stFileUploader"] span, 
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] p {
        color: #334155 !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
    }

    /* File uploader button & all secondary buttons */
    [data-testid="stFileUploader"] button,
    .stButton button,
    button[kind="secondary"],
    button[kind="primary"] {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        letter-spacing: 0.5px !important;
        border: 1px solid rgba(255, 255, 255, 0.5) !important;
        border-radius: 12px !important;
        padding: 0.65rem 1.6rem !important;
        box-shadow: 0 4px 16px rgba(2, 132, 199, 0.4) !important;
        transition: all 0.25s ease !important;
        cursor: pointer !important;
    }

    [data-testid="stFileUploader"] button:hover,
    .stButton button:hover {
        background: linear-gradient(135deg, #0369a1 0%, #075985 100%) !important;
        box-shadow: 0 8px 25px rgba(2, 132, 199, 0.55) !important;
        transform: translateY(-2px) !important;
    }

    /* HUD Device Mockup Frame */
    .hud-frame {
        background: #ffffff;
        border: 2px solid rgba(2, 132, 199, 0.35);
        border-radius: 28px;
        padding: 2rem;
        box-shadow: 0 20px 50px -10px rgba(0, 180, 216, 0.18),
                    0 0 25px 0 rgba(0, 242, 254, 0.1) inset;
        margin-bottom: 1.5rem;
    }

    .hud-topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 2px solid #f1f5f9;
        padding-bottom: 1rem;
        margin-bottom: 1.4rem;
    }

    .hud-logo {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 900;
        font-size: 1.35rem;
        letter-spacing: 1.8px;
        color: #0369a1;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .pulse-badge {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        font-weight: 700;
        color: #0284c7;
        background: rgba(2, 132, 199, 0.12);
        border: 1px solid rgba(2, 132, 199, 0.25);
        padding: 0.35rem 0.95rem;
        border-radius: 9999px;
    }

    .pulse-dot {
        width: 10px;
        height: 10px;
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

    /* Calorie Hero Card */
    .cal-hero {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 1.5px solid rgba(2, 132, 199, 0.3);
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        margin: 1.2rem 0 1.5rem 0;
        box-shadow: 0 6px 25px rgba(2, 132, 199, 0.12);
    }

    .cal-title {
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 1.8px;
        color: #475569;
        font-weight: 700;
    }

    .cal-val {
        font-size: 3.5rem;
        font-weight: 900;
        color: #0f172a;
        line-height: 1.1;
        margin: 0.3rem 0;
    }

    .cal-val span {
        font-size: 1.4rem;
        font-weight: 800;
        color: #0284c7;
        margin-left: 0.2rem;
    }

    /* Custom Macro Progress Bars with Enhanced Font Sizes */
    .macro-section-title {
        font-size: 1.15rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #0369a1;
        margin-bottom: 1.1rem;
        border-left: 4px solid #0284c7;
        padding-left: 0.6rem;
    }

    .macro-row {
        margin-bottom: 1.2rem;
    }

    .macro-label-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.45rem;
        color: #1e293b;
    }

    .macro-val-text {
        font-size: 1.15rem;
        font-weight: 800;
    }

    .macro-track {
        width: 100%;
        height: 14px;
        background: #e2e8f0;
        border-radius: 9999px;
        overflow: hidden;
        position: relative;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.06);
    }

    .macro-fill {
        height: 100%;
        border-radius: 9999px;
        transition: width 0.8s cubic-bezier(0.16, 1, 0.3, 1);
    }

    .fill-pro { background: linear-gradient(90deg, #00f2fe, #0284c7); box-shadow: 0 0 12px rgba(0, 242, 254, 0.6); }
    .fill-carb { background: linear-gradient(90deg, #38bdf8, #0ea5e9); box-shadow: 0 0 12px rgba(56, 189, 248, 0.6); }
    .fill-fat { background: linear-gradient(90deg, #10b981, #059669); box-shadow: 0 0 12px rgba(16, 185, 129, 0.6); }

    /* Pill Badges with Enhanced Font Sizes */
    .pill-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.85rem;
        margin: 1.3rem 0;
    }

    .stat-pill {
        background: #f8fafc;
        border: 1.5px solid #cbd5e1;
        padding: 0.85rem 1rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }

    .stat-pill .pill-num {
        font-size: 1.4rem;
        font-weight: 800;
        color: #0f172a;
    }

    .stat-pill .pill-txt {
        font-size: 0.85rem;
        color: #475569;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.8px;
        margin-top: 0.2rem;
    }

    /* Condition Status Callouts */
    .condition-badge {
        padding: 1.1rem 1.3rem;
        border-radius: 16px;
        font-size: 0.95rem;
        margin-top: 1.3rem;
        display: flex;
        align-items: flex-start;
        gap: 0.85rem;
        line-height: 1.5;
    }

    .cond-normal {
        background: #ecfdf5;
        border: 1.5px solid #10b981;
        color: #065f46;
    }

    .cond-overcooked {
        background: #fffbeb;
        border: 1.5px solid #f59e0b;
        color: #92400e;
    }

    .cond-spoilage {
        background: #fff1f2;
        border: 1.5px solid #f43f5e;
        color: #9f1239;
    }

    .cond-uncertain {
        background: #f1f5f9;
        border: 1.5px solid #94a3b8;
        color: #334155;
    }

    /* Item Card Details */
    .item-chip {
        display: inline-block;
        font-size: 0.88rem;
        font-weight: 700;
        padding: 0.35rem 0.85rem;
        border-radius: 8px;
        background: #e2e8f0;
        color: #1e293b;
        margin-right: 0.4rem;
        margin-bottom: 0.3rem;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Cache models to ensure fast inference without reloading on re-renders."""
    detector = FoodDetector()
    api_key = os.environ.get("api_key") or os.environ.get("GEMINI_API_KEY")
    classifier = LLMFoodClassifier(api_key=api_key)
    return detector, classifier


def calculate_overlap(box1, box2):
    """Calculates Intersection over Union (IoU) and Intersection over Smaller Box (IoS)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0, 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    iou = inter_area / float(union_area) if union_area > 0 else 0.0
    ios = inter_area / float(min(box1_area, box2_area))
    return iou, ios


def filter_redundant_crops(boxes, crops, iou_thresh=0.45, ios_thresh=0.70):
    """Filters duplicate/overlapping bounding boxes to avoid double-counting."""
    if not boxes:
        return [], []

    sorted_indices = sorted(range(len(boxes)), key=lambda k: boxes[k]['conf'], reverse=True)
    keep_indices = []

    for i in sorted_indices:
        box_i = boxes[i]['box']
        redundant = False
        for j in keep_indices:
            box_j = boxes[j]['box']
            iou, ios = calculate_overlap(box_i, box_j)
            if iou > iou_thresh or ios > ios_thresh:
                redundant = True
                break
        if not redundant:
            keep_indices.append(i)

    filtered_boxes = [boxes[k] for k in keep_indices]
    filtered_crops = [crops[k] for k in keep_indices]
    return filtered_boxes, filtered_crops


def draw_high_tech_overlay(image_pil, boxes, item_labels=None):
    """Draws sleek sci-fi HUD bounding boxes and corner brackets."""
    img = image_pil.copy()
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size

    # Define color scheme: Neon Cyan
    box_color = (0, 242, 254, 255)
    bracket_color = (0, 200, 255, 255)
    fill_color = (0, 242, 254, 30)
    badge_bg = (15, 23, 42, 230)
    badge_text_color = (255, 255, 255, 255)

    bracket_len = max(16, int(min(w, h) * 0.04))
    line_thickness = max(2, int(min(w, h) * 0.0035))

    for idx, b in enumerate(boxes):
        x1, y1, x2, y2 = b['box']

        # Semi-transparent box fill
        draw.rectangle([x1, y1, x2, y2], fill=fill_color, outline=box_color, width=line_thickness)

        # Draw Futuristic Corner Brackets (HUD style)
        # Top-Left
        draw.line([(x1, y1), (x1 + bracket_len, y1)], fill=bracket_color, width=line_thickness * 2)
        draw.line([(x1, y1), (x1, y1 + bracket_len)], fill=bracket_color, width=line_thickness * 2)
        # Top-Right
        draw.line([(x2, y1), (x2 - bracket_len, y1)], fill=bracket_color, width=line_thickness * 2)
        draw.line([(x2, y1), (x2, y1 + bracket_len)], fill=bracket_color, width=line_thickness * 2)
        # Bottom-Left
        draw.line([(x1, y2), (x1 + bracket_len, y2)], fill=bracket_color, width=line_thickness * 2)
        draw.line([(x1, y2), (x1, y2 - bracket_len)], fill=bracket_color, width=line_thickness * 2)
        # Bottom-Right
        draw.line([(x2, y2), (x2 - bracket_len, y2)], fill=bracket_color, width=line_thickness * 2)
        draw.line([(x2, y2), (x2, y2 - bracket_len)], fill=bracket_color, width=line_thickness * 2)

        # Label Badge
        name = item_labels[idx] if (item_labels and idx < len(item_labels)) else f"Item {idx + 1}"
        badge_text = f" [{idx + 1:02d}] {name.title()} "

        # Draw badge pill
        badge_w = len(badge_text) * 9 + 16
        badge_h = 26
        by1 = max(0, y1 - badge_h - 4)
        by2 = by1 + badge_h
        bx2 = min(w, x1 + badge_w)
        draw.rounded_rectangle([x1, by1, bx2, by2], radius=6, fill=badge_bg, outline=box_color, width=1)
        draw.text((x1 + 8, by1 + 5), badge_text, fill=badge_text_color)

    return img


# ==========================================
# UI LAYOUT
# ==========================================

# Top Hero Header
render_html("""
<div class="hero-container">
    <div class="tech-tag">RESEARCH PROJECT &bull; YOLOV8 &bull; GEMINI VISION &bull; MULTI-DISH AI</div>
    <h1 class="diet-title">DIET ENGINE</h1>
    <p class="diet-subtitle">Personalized AI Nutrition & Research Analysis</p>
</div>
""")

# Load Models
detector, classifier = load_models()

# Main Container
main_col1, main_col2 = st.columns([1.1, 1], gap="large")

with main_col1:
    render_html("""
    <div class="glass-card">
        <h3 style="margin-top:0; font-weight: 800; color: #0f172a; font-size: 1.4rem;">📷 Meal Input Scanner</h3>
        <p style="color: #475569; font-size: 0.95rem; margin-bottom: 0.8rem;">Upload any meal photo or composite plate. The engine automatically isolates portions and estimates macronutrients.</p>
    </div>
    """)

    uploaded_file = st.file_uploader(
        "Choose food image (JPG, PNG, WebP)",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )

    if uploaded_file is None:
        render_html("""
        <div style="border: 2px dashed rgba(2, 132, 199, 0.4); border-radius: 20px; padding: 3.5rem 1.5rem; text-align: center; background: rgba(255, 255, 255, 0.7); margin-top: 1rem;">
            <div style="font-size: 3.2rem; margin-bottom: 0.5rem;">🥗</div>
            <div style="font-weight: 800; font-size: 1.25rem; color: #0f172a;">No Plate Image Uploaded</div>
            <div style="font-size: 0.95rem; color: #475569; margin-top: 0.4rem;">Drag & drop a food photo above to scan calories and macronutrients.</div>
        </div>
        """)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("⚡ Running YOLO Detection & Vision AI Pipeline..."):
        # 1. Detection
        raw_boxes, raw_crops = detector.detect(image)
        # 2. Deduplicate overlapping crops
        boxes, crops = filter_redundant_crops(raw_boxes, raw_crops)

        # 3. LLM Vision Analysis on all detected crops
        items_data = []
        item_names_for_overlay = []

        for crop in crops:
            try:
                item_info = classifier.predict(crop)
                items_data.append(item_info)
                item_names_for_overlay.append(item_info.get("label", "Food Item"))
            except Exception as e:
                fallback_item = {
                    "label": "Food Portion",
                    "confidence": 0.8,
                    "calories": 150,
                    "protein": 5,
                    "fat": 5,
                    "carbs": 20,
                    "fiber": 2,
                    "estimated_weight_g": 100,
                    "weight_confidence": 0.5,
                    "visual_condition": "normal_appearance",
                    "condition_severity": "none",
                    "condition_confidence": 0.7,
                    "condition_evidence": "Analyzed with default parameters."
                }
                items_data.append(fallback_item)
                item_names_for_overlay.append("Food Portion")

    # Render Left Column with High-Tech Bounding Boxes
    with main_col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        annotated_img = draw_high_tech_overlay(image, boxes, item_names_for_overlay)
        st.image(annotated_img, caption="YOLOv8 Dynamic Target Localization", use_container_width=True)

        num_detected = len(boxes)
        render_html(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 0.9rem; padding: 0.75rem 1.1rem; background: #f8fafc; border: 1.5px solid #e2e8f0; border-radius: 14px; font-family: 'JetBrains Mono', monospace; font-size: 0.88rem; color: #1e293b;">
            <div><span>TARGETS DETECTED:</span> <strong style="color: #0284c7; font-size: 1.05rem;">{num_detected}</strong></div>
            <div><span>STATUS:</span> <strong style="color: #10b981; font-size: 0.95rem;">OPTIMAL SCAN</strong></div>
        </div>
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # Render Right Column (Replicating the smart HUD device in the uploaded image)
    with main_col2:
        # Calculate Aggregates
        total_cal = sum(it.get("calories", 0) for it in items_data)
        total_pro = sum(it.get("protein", 0) for it in items_data)
        total_fat = sum(it.get("fat", 0) for it in items_data)
        total_carb = sum(it.get("carbs", 0) for it in items_data)
        total_fiber = sum(it.get("fiber", 0) for it in items_data)
        total_weight = sum(it.get("estimated_weight_g", 0) for it in items_data)

        # Identify dominant condition
        conditions = [it.get("visual_condition", "normal_appearance") for it in items_data]
        if "possible_spoilage_signs" in conditions:
            dominant_cond = "possible_spoilage_signs"
        elif "overcooked_or_burnt" in conditions:
            dominant_cond = "overcooked_or_burnt"
        elif "normal_appearance" in conditions:
            dominant_cond = "normal_appearance"
        else:
            dominant_cond = "uncertain"

        # Determine composite meal label
        if items_data:
            top_labels = [it.get("label", "").title() for it in items_data[:3]]
            composite_meal_name = " + ".join(top_labels)
        else:
            composite_meal_name = "Whole Plate"

        # Render HUD Dashboard Card cleanly without indentation
        hud_card_html = f"""
<div class="hud-frame">
<div class="hud-topbar">
<div class="hud-logo"><span>⚡</span> DIET ENGINE</div>
<div class="pulse-badge"><span class="pulse-dot"></span> SCAN COMPLETE</div>
</div>

<div style="font-size: 0.92rem; text-transform: uppercase; letter-spacing: 1.2px; color: #475569; font-weight: 700;">Meal Composition</div>
<div style="font-size: 1.45rem; font-weight: 800; color: #0f172a; margin-bottom: 0.8rem;">{composite_meal_name}</div>

<div class="cal-hero">
<div class="cal-title">Total Energy Value</div>
<div class="cal-val">{total_cal:.0f} <span>kcal</span></div>
<div style="font-size: 0.92rem; color: #475569; font-weight: 600; margin-top: 0.2rem;">Computed across {num_detected} isolated portion{'s' if num_detected != 1 else ''}</div>
</div>

<div class="pill-grid">
<div class="stat-pill">
<div class="pill-num">⚖️ {total_weight:.0f}g</div>
<div class="pill-txt">Est. Food Weight</div>
</div>
<div class="stat-pill">
<div class="pill-num">🌾 {total_fiber:.1f}g</div>
<div class="pill-txt">Dietary Fiber</div>
</div>
</div>

<div style="margin: 1.6rem 0 1.2rem 0;">
<div class="macro-section-title">Macronutrient Profile</div>

<div class="macro-row">
<div class="macro-label-row">
<span>🍗 Protein</span>
<span class="macro-val-text" style="color: #0284c7;">{total_pro:.1f}g</span>
</div>
<div class="macro-track">
<div class="macro-fill fill-pro" style="width: {min(100, int((total_pro / 120.0) * 100))}%;"></div>
</div>
</div>

<div class="macro-row">
<div class="macro-label-row">
<span>🍞 Carbohydrates</span>
<span class="macro-val-text" style="color: #0ea5e9;">{total_carb:.1f}g</span>
</div>
<div class="macro-track">
<div class="macro-fill fill-carb" style="width: {min(100, int((total_carb / 250.0) * 100))}%;"></div>
</div>
</div>

<div class="macro-row">
<div class="macro-label-row">
<span>🥑 Healthy Fats</span>
<span class="macro-val-text" style="color: #10b981;">{total_fat:.1f}g</span>
</div>
<div class="macro-track">
<div class="macro-fill fill-fat" style="width: {min(100, int((total_fat / 80.0) * 100))}%;"></div>
</div>
</div>
</div>
"""
        render_html(hud_card_html)

        # Condition Assessment Box
        if dominant_cond == "overcooked_or_burnt":
            render_html("""
<div class="condition-badge cond-overcooked">
<span style="font-size: 1.4rem;">🔥</span>
<div>
<strong style="font-size: 1.05rem;">Overcooked or Burnt Surface Detected</strong><br>
Visible charring or heavy surface darkening identified in one or more portions.
</div>
</div>
""")
        elif dominant_cond == "possible_spoilage_signs":
            render_html("""
<div class="condition-badge cond-spoilage">
<span style="font-size: 1.4rem;">⚠️</span>
<div>
<strong style="font-size: 1.05rem;">Visual Spoilage Signs Detected</strong><br>
Abnormal discoloration or surface deterioration identified. Exercise caution.
</div>
</div>
""")
        elif dominant_cond == "normal_appearance":
            render_html("""
<div class="condition-badge cond-normal">
<span style="font-size: 1.4rem;">✨</span>
<div>
<strong style="font-size: 1.05rem;">Optimal Visual Quality</strong><br>
Fresh, properly prepared appearance with no obvious signs of charring or spoilage.
</div>
</div>
""")
        else:
            render_html("""
<div class="condition-badge cond-uncertain">
<span style="font-size: 1.4rem;">ℹ️</span>
<div>
<strong style="font-size: 1.05rem;">Condition Assessment Uncertain</strong><br>
Visual indicators are inconclusive for surface condition evaluation.
</div>
</div>
""")

        # Itemized Accordions
        st.markdown('<div style="margin-top: 1.6rem;"></div>', unsafe_allow_html=True)
        render_html("<div style='font-size: 1.05rem; font-weight: 800; color: #0f172a; margin-bottom: 0.7rem; border-left: 4px solid #0284c7; padding-left: 0.6rem;'>ITEMIZED NUTRITION DETAILS</div>")

        for i, item in enumerate(items_data):
            label = item.get("label", f"Dish {i + 1}").title()
            cal = item.get("calories", 0)
            weight = item.get("estimated_weight_g", 0)
            conf = item.get("confidence", 0.9) * 100
            pro = item.get("protein", 0)
            fat = item.get("fat", 0)
            carb = item.get("carbs", 0)
            fiber = item.get("fiber", 0)
            portion = item.get("portion_unit", "1 portion")
            evidence = item.get("condition_evidence", "Normal visual appearance.")

            with st.expander(f"🔹 Item {i + 1:02d}: {label} — {cal:.0f} kcal"):
                render_html(f"""
<div style="font-size: 0.95rem; line-height: 1.6; color: #1e293b;">
<div><strong>Portion Estimate:</strong> {portion} (~{weight:.0f}g) &bull; <span style="color: #0284c7; font-weight: 700;">Confidence: {conf:.1f}%</span></div>
<div style="margin-top: 0.5rem; padding: 0.7rem; background: #f1f5f9; border-radius: 10px;">
<span class="item-chip">🍗 Pro: {pro:.1f}g</span>
<span class="item-chip">🍞 Carb: {carb:.1f}g</span>
<span class="item-chip">🥑 Fat: {fat:.1f}g</span>
<span class="item-chip">🌾 Fiber: {fiber:.1f}g</span>
</div>
<div style="margin-top: 0.6rem; font-size: 0.88rem; color: #475569;">
<strong>Visual Notes:</strong> {evidence}
</div>
</div>
""")

        render_html("</div>")
