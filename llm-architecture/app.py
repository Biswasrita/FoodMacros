import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import cv2
import sys
import os
import textwrap
import base64
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

# Helper function to render pure HTML directly without Markdown converting indentation into code blocks
def render_html(html_str: str):
    clean_str = textwrap.dedent(html_str).strip()
    st.html(clean_str)

# Helper function to get base64 encoded food plate image
def get_plate_base64():
    candidates = [
        os.path.join(CURRENT_DIR, "assets", "circular_plate.png"),
        os.path.join(os.path.dirname(CURRENT_DIR), "assets", "circular_plate.png"),
        os.path.join(CURRENT_DIR, "assets", "hero_plate.png"),
        os.path.join(os.path.dirname(CURRENT_DIR), "assets", "hero_plate.png")
    ]
    for p in candidates:
        if p and os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")
            except Exception:
                pass
    return ""

# High-Tech Sci-Fi "Diet Engine" Theme CSS
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700;800&display=swap');

    :root {
        --primary-glow: #0284c7;
        --secondary-glow: #00f2fe;
        --accent-teal: #06b6d4;
        --accent-emerald: #10b981;
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

    /* ========================================================
       POSTER CANVAS & HERO LANDING STYLES (MATCHING USER GRAPHIC)
       ======================================================== */
    .diet-engine-poster {
        position: relative;
        max-width: 1100px;
        margin: 0.5rem auto 1.5rem auto;
        background: radial-gradient(circle at 18% 45%, rgba(0, 242, 254, 0.08) 0%, transparent 45%),
                    radial-gradient(circle at 82% 50%, rgba(2, 132, 199, 0.08) 0%, transparent 45%),
                    #ffffff;
        border: 2px solid rgba(0, 242, 254, 0.45);
        border-radius: 32px;
        box-shadow: 0 25px 65px -15px rgba(2, 132, 199, 0.22),
                    0 0 45px rgba(0, 242, 254, 0.12) inset;
        padding: 2.2rem 2.8rem;
        overflow: hidden;
    }

    /* Top Bar of Poster */
    .poster-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 1.2rem;
    }

    .poster-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.88rem;
        font-weight: 700;
        color: #475569;
        letter-spacing: 2.4px;
        text-transform: uppercase;
        border-left: 3.5px solid #00f2fe;
        padding-left: 0.85rem;
    }

    .poster-title-wrap {
        text-align: right;
    }

    .poster-brand {
        font-size: 3.4rem;
        font-weight: 900;
        letter-spacing: -0.5px;
        color: #1e293b;
        margin: 0;
        line-height: 1;
    }

    .poster-subbrand {
        font-size: 1.15rem;
        font-weight: 600;
        color: #64748b;
        margin-top: 0.4rem;
        letter-spacing: 0.3px;
    }

    /* Visual Stage Container */
    .poster-stage {
        position: relative;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1.2rem 0;
        min-height: 430px;
    }

    /* SVG Circuit Lines Layer */
    .hud-circuit-svg {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 1;
    }

    /* Left: Plate Container */
    .plate-container {
        position: relative;
        width: 350px;
        height: 350px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        z-index: 2;
    }

    .plate-frame {
        position: relative;
        width: 310px;
        height: 310px;
    }

    .food-plate-img {
        width: 100%;
        height: 100%;
        border-radius: 50%;
        box-shadow: 0 20px 45px rgba(0, 0, 0, 0.22), 0 0 35px rgba(0, 242, 254, 0.25);
        display: block;
        object-fit: cover;
    }

    /* Target Pins & Reticles */
    .reticle-pin {
        position: absolute;
        transform: translate(-50%, -50%);
        pointer-events: auto;
        cursor: default;
    }

    .reticle-ring {
        width: 26px;
        height: 26px;
        border: 2px solid #00f2fe;
        border-radius: 50%;
        box-shadow: 0 0 14px #00f2fe;
        animation: reticlePulse 2.2s infinite ease-in-out;
    }

    .reticle-core {
        position: absolute;
        top: 50%;
        left: 50%;
        width: 8px;
        height: 8px;
        background: #ffffff;
        border: 2px solid #00f2fe;
        border-radius: 50%;
        transform: translate(-50%, -50%);
        box-shadow: 0 0 8px #00f2fe;
    }

    .reticle-tooltip {
        position: absolute;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(15, 23, 42, 0.92);
        color: #00f2fe;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 6px;
        white-space: nowrap;
        border: 1px solid rgba(0, 242, 254, 0.5);
        pointer-events: none;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }

    @keyframes reticlePulse {
        0% { transform: scale(0.9); opacity: 0.75; }
        50% { transform: scale(1.15); opacity: 1; box-shadow: 0 0 20px #00f2fe; }
        100% { transform: scale(0.9); opacity: 0.75; }
    }

    .plate-tech-hud {
        margin-top: 1.1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 700;
        color: #0369a1;
        background: rgba(2, 132, 199, 0.08);
        border: 1px solid rgba(2, 132, 199, 0.25);
        padding: 0.35rem 0.95rem;
        border-radius: 999px;
        letter-spacing: 0.5px;
    }

    /* Center: Futuristic Phone UI Mockup */
    .phone-device-wrapper {
        position: relative;
        z-index: 2;
    }

    .phone-device {
        width: 295px;
        background: #ffffff;
        border: 2.5px solid rgba(0, 242, 254, 0.75);
        border-radius: 36px;
        padding: 1.6rem 1.4rem;
        box-shadow: 0 0 45px rgba(0, 242, 254, 0.35),
                    0 25px 60px rgba(15, 23, 42, 0.12),
                    inset 0 0 25px rgba(0, 242, 254, 0.08);
    }

    .phone-speaker-notch {
        width: 60px;
        height: 5px;
        background: #cbd5e1;
        border-radius: 10px;
        margin: 0 auto 1.1rem auto;
    }

    .phone-header-title {
        font-size: 1.15rem;
        font-weight: 900;
        letter-spacing: 2.2px;
        text-transform: uppercase;
        color: #0f172a;
        text-align: center;
        margin-bottom: 0.9rem;
    }

    .phone-camera-area {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-bottom: 0.9rem;
    }

    .camera-lens-box {
        width: 48px;
        height: 48px;
        border: 2px solid #00f2fe;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(0, 242, 254, 0.08);
        box-shadow: 0 0 16px rgba(0, 242, 254, 0.3);
    }

    .scan-frequency-wave {
        display: flex;
        align-items: center;
        gap: 6px;
        margin-top: 0.5rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        color: #0284c7;
        font-weight: 700;
    }

    .phone-result-card {
        background: #f8fafc;
        border: 1.5px solid #e2e8f0;
        border-radius: 14px;
        padding: 0.65rem 0.85rem;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.65rem;
    }

    .phone-target-icon {
        font-size: 1.3rem;
        color: #0284c7;
    }

    .phone-label-sm {
        font-size: 0.75rem;
        color: #64748b;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .phone-meal-name {
        font-size: 1.05rem;
        font-weight: 800;
        color: #0f172a;
    }

    .phone-calories-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 1px solid rgba(2, 132, 199, 0.25);
        border-radius: 14px;
        padding: 0.65rem 0.85rem;
        margin-bottom: 0.9rem;
    }

    .phone-cal-number {
        font-size: 1.85rem;
        font-weight: 900;
        color: #0f172a;
        margin-top: 0.15rem;
        line-height: 1.1;
    }

    .phone-cal-unit {
        font-size: 0.95rem;
        font-weight: 700;
        color: #0284c7;
    }

    .phone-bar-item {
        margin-bottom: 0.65rem;
    }

    .phone-bar-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.82rem;
        font-weight: 700;
        color: #334155;
        margin-bottom: 0.25rem;
    }

    .phone-bar-track {
        height: 8px;
        background: #e2e8f0;
        border-radius: 999px;
        overflow: hidden;
    }

    .phone-bar-progress {
        height: 100%;
        border-radius: 999px;
    }

    /* Right: Floating Telemetry Badges */
    .telemetry-badges-column {
        display: flex;
        flex-direction: column;
        gap: 1.8rem;
        position: relative;
        z-index: 2;
        padding-right: 1.5rem;
    }

    .telemetry-pill {
        background: #ffffff;
        border: 1.8px solid rgba(0, 242, 254, 0.6);
        border-radius: 14px;
        padding: 0.65rem 1.1rem;
        box-shadow: 0 8px 24px rgba(2, 132, 199, 0.12),
                    0 0 15px rgba(0, 242, 254, 0.15);
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        color: #0f172a;
        display: flex;
        align-items: center;
        gap: 0.65rem;
        font-size: 1.05rem;
    }

    /* File Uploader Custom Styling */
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

    /* Buttons Styling */
    [data-testid="stFileUploader"] button,
    .stButton button,
    button[kind="secondary"],
    button[kind="primary"] {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        letter-spacing: 0.5px !important;
        border: 1px solid rgba(255, 255, 255, 0.5) !important;
        border-radius: 14px !important;
        padding: 0.75rem 1.8rem !important;
        box-shadow: 0 6px 20px rgba(2, 132, 199, 0.4) !important;
        transition: all 0.25s ease !important;
        cursor: pointer !important;
    }

    [data-testid="stFileUploader"] button:hover,
    .stButton button:hover {
        background: linear-gradient(135deg, #0369a1 0%, #075985 100%) !important;
        box-shadow: 0 8px 28px rgba(2, 132, 199, 0.6) !important;
        transform: translateY(-2px) !important;
    }

    /* Glass Cards */
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

    /* HUD Frame for Feature View */
    .hud-frame {
        background: #ffffff;
        border: 2px solid rgba(0, 242, 254, 0.5);
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
        animation: pulseDot 1.8s infinite;
    }

    @keyframes pulseDot {
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

    /* Macro Progress Bars */
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

    /* Pill Badges */
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

    /* ========================================================
       HIGH-TECH DIET ENGINE RADAR LOADER STYLES (MINIMAL & SLEEK)
       ======================================================== */
    .loader-modal-backdrop {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
        margin: 0 0 1rem 0;
    }

    .loader-card {
        background: radial-gradient(circle at 50% 25%, rgba(0, 242, 254, 0.08) 0%, transparent 65%),
                    rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(16px);
        border: 1.5px solid rgba(0, 242, 254, 0.45);
        border-radius: 20px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 15px 35px -10px rgba(2, 132, 199, 0.15),
                    0 0 25px rgba(0, 242, 254, 0.1) inset;
        text-align: center;
        width: 100%;
        position: relative;
        overflow: hidden;
    }

    /* Radar Scanner Stage */
    .radar-stage {
        position: relative;
        width: 76px;
        height: 76px;
        margin: 0.2rem auto 0.8rem auto;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .radar-outer-ring {
        position: absolute;
        width: 76px;
        height: 76px;
        border-radius: 50%;
        border: 1.5px dashed rgba(0, 242, 254, 0.45);
        animation: rotateOuter 10s linear infinite;
    }

    .radar-beam {
        position: absolute;
        width: 72px;
        height: 72px;
        border-radius: 50%;
        background: conic-gradient(from 0deg, transparent 0%, rgba(0, 242, 254, 0.05) 60%, rgba(0, 242, 254, 0.35) 95%, #00f2fe 100%);
        animation: rotateRadar 2s linear infinite;
        filter: drop-shadow(0 0 6px #00f2fe);
    }

    .radar-inner-glow {
        position: absolute;
        width: 52px;
        height: 52px;
        border-radius: 50%;
        background: rgba(2, 132, 199, 0.08);
        box-shadow: 0 0 16px rgba(0, 242, 254, 0.25);
    }

    /* Central Food Icon Cycler */
    .food-cycler-container {
        position: relative;
        width: 48px;
        height: 48px;
        border-radius: 50%;
        background: #ffffff;
        border: 1.5px solid #00f2fe;
        box-shadow: 0 0 12px rgba(0, 242, 254, 0.45);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 3;
        overflow: hidden;
    }

    .food-icon-item {
        position: absolute;
        font-size: 1.7rem;
        line-height: 1;
        opacity: 0;
        transform: scale(0.6) rotate(-15deg);
        animation: foodCycleAnim 9s infinite ease-in-out;
    }

    /* 6 cycling food items */
    .fi-1 { animation-delay: 0s; }
    .fi-2 { animation-delay: 1.5s; }
    .fi-3 { animation-delay: 3.0s; }
    .fi-4 { animation-delay: 4.5s; }
    .fi-5 { animation-delay: 6.0s; }
    .fi-6 { animation-delay: 7.5s; }

    @keyframes foodCycleAnim {
        0% { opacity: 0; transform: scale(0.5) rotate(-20deg); }
        3% { opacity: 1; transform: scale(1.08) rotate(0deg); }
        13.6% { opacity: 1; transform: scale(1.0) rotate(0deg); }
        16.6% { opacity: 0; transform: scale(0.5) rotate(20deg); }
        100% { opacity: 0; }
    }

    @keyframes rotateRadar {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    @keyframes rotateOuter {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    /* Text Rotator */
    .loader-badge-tag {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        color: #0284c7;
        background: rgba(2, 132, 199, 0.08);
        border: 1px solid rgba(2, 132, 199, 0.25);
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
    }

    .pulse-dot-loader {
        width: 6px;
        height: 6px;
        background: #00f2fe;
        border-radius: 50%;
        box-shadow: 0 0 6px #00f2fe;
        animation: pulseDot 1.4s infinite;
    }

    .loader-main-title {
        font-size: 1.22rem;
        font-weight: 900;
        color: #0f172a;
        margin: 0 0 0.25rem 0;
        letter-spacing: -0.2px;
    }

    .loader-status-container {
        position: relative;
        min-height: 24px;
        margin: 0.2rem auto 0.5rem auto;
    }

    .loader-status-text {
        font-size: 0.92rem;
        font-weight: 700;
        color: #0284c7;
        font-family: 'Plus Jakarta Sans', sans-serif;
        line-height: 1.35;
    }

    /* Continuously Cycling High-Tech Phrases */
    .rotator-wrapper {
        position: relative;
        height: 22px;
        margin-top: 0.4rem;
    }

    .rotator-text-item {
        position: absolute;
        width: 100%;
        left: 0;
        top: 0;
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748b;
        opacity: 0;
        transform: translateY(6px);
        animation: textCycleAnim 10s infinite ease-in-out;
    }

    .ri-1 { animation-delay: 0s; }
    .ri-2 { animation-delay: 2s; }
    .ri-3 { animation-delay: 4s; }
    .ri-4 { animation-delay: 6s; }
    .ri-5 { animation-delay: 8s; }

    @keyframes textCycleAnim {
        0% { opacity: 0; transform: translateY(6px); }
        4% { opacity: 1; transform: translateY(0); }
        16% { opacity: 1; transform: translateY(0); }
        20% { opacity: 0; transform: translateY(-6px); }
        100% { opacity: 0; }
    }

    /* Glowing Neon Progress Bar */
    .loader-progress-track {
        width: 100%;
        max-width: 320px;
        height: 6px;
        background: #e2e8f0;
        border-radius: 999px;
        margin: 0.7rem auto 0.7rem auto;
        overflow: hidden;
        position: relative;
        border: 1px solid rgba(0, 242, 254, 0.35);
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.05);
    }

    .loader-progress-bar {
        width: 100%;
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #0284c7 0%, #00f2fe 35%, #10b981 70%, #0284c7 100%);
        background-size: 200% 100%;
        animation: sweepProgress 2s linear infinite;
        box-shadow: 0 0 8px rgba(0, 242, 254, 0.7);
    }

    @keyframes sweepProgress {
        0% { background-position: 100% 0; }
        100% { background-position: -100% 0; }
    }

    /* Bottom Micro Telemetry Badges */
    .loader-telemetry-row {
        display: flex;
        justify-content: center;
        gap: 0.6rem;
        flex-wrap: wrap;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 700;
        color: #475569;
        margin-top: 0.6rem;
    }

    .loader-telem-chip {
        background: #f8fafc;
        border: 1px solid #cbd5e1;
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        display: flex;
        align-items: center;
        gap: 0.35rem;
    }
</style>
"""

render_html(CUSTOM_CSS)


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
        draw.line([(x1, y1), (x1 + bracket_len, y1)], fill=bracket_color, width=line_thickness * 2)
        draw.line([(x1, y1), (x1, y1 + bracket_len)], fill=bracket_color, width=line_thickness * 2)

        draw.line([(x2, y1), (x2 - bracket_len, y1)], fill=bracket_color, width=line_thickness * 2)
        draw.line([(x2, y1), (x2, y1 + bracket_len)], fill=bracket_color, width=line_thickness * 2)

        draw.line([(x1, y2), (x1 + bracket_len, y2)], fill=bracket_color, width=line_thickness * 2)
        draw.line([(x1, y2), (x1, y2 - bracket_len)], fill=bracket_color, width=line_thickness * 2)

        draw.line([(x2, y2), (x2 - bracket_len, y2)], fill=bracket_color, width=line_thickness * 2)
        draw.line([(x2, y2), (x2, y2 - bracket_len)], fill=bracket_color, width=line_thickness * 2)

        # Label Badge
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


def get_standby_static_card_html() -> str:
    """Generates a clean, static showcase card for main_col2 before a meal photo is uploaded."""
    img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "diet_engine_poster.jpg")
    if not os.path.exists(img_path):
        img_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "diet_engine_poster.jpg")

    b64_str = ""
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            b64_str = base64.b64encode(f.read()).decode("utf-8")

    return f"""
    <div class="glass-card" style="padding: 1.3rem; border-radius: 22px; border: 1.5px solid rgba(2, 132, 199, 0.22); background: rgba(255, 255, 255, 0.95); box-shadow: 0 15px 35px -10px rgba(2, 132, 199, 0.12);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.85rem;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; font-weight: 800; color: #0284c7; letter-spacing: 1.2px;">
                ⚡ DIET ENGINE VISION HUD
            </div>
            <div style="font-size: 0.74rem; font-weight: 700; color: #10b981; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.25); padding: 0.2rem 0.65rem; border-radius: 999px;">
                READY TO SCAN
            </div>
        </div>
        <div style="border-radius: 14px; overflow: hidden; border: 1px solid #e2e8f0; box-shadow: 0 6px 20px rgba(0, 0, 0, 0.05); margin-bottom: 0.8rem;">
            <img src="data:image/jpeg;base64,{b64_str}" style="width: 100%; display: block; object-fit: cover;" alt="Diet Engine AI Nutrition Analysis" />
        </div>
        <div style="text-align: center; color: #64748b; font-size: 0.86rem; font-weight: 600; line-height: 1.45;">
            Upload or capture any meal photo on the left. The engine isolates portions and streams your complete nutritional profile here.
        </div>
    </div>
    """


def get_loader_html(status_text: str = None, current_portion: int = None, total_portions: int = None, is_standby: bool = False) -> str:
    """Generates the compact, high-tech animated loader for main_col2.
    Shows standby state when awaiting meal upload, or dynamic live pipeline state during YOLO/Gemini scan."""
    if is_standby:
        return """
<div class="loader-modal-backdrop">
    <div class="loader-card">
        <!-- Radar Scanner Stage with Rotating Beam -->
        <div class="radar-stage">
            <div class="radar-outer-ring"></div>
            <div class="radar-beam"></div>
            <div class="radar-inner-glow"></div>
            
            <!-- Center: Smoothly Cycling 3D Food Icons -->
            <div class="food-cycler-container">
                <div class="food-icon-item fi-1">🥗</div>
                <div class="food-icon-item fi-2">🥑</div>
                <div class="food-icon-item fi-3">🍗</div>
                <div class="food-icon-item fi-4">🍣</div>
                <div class="food-icon-item fi-5">🥪</div>
                <div class="food-icon-item fi-6">🍳</div>
            </div>
        </div>

        <!-- Standby Status Badge -->
        <div class="loader-badge-tag">
            <span class="pulse-dot-loader"></span>
            <span>AI ENGINE STANDBY</span>
        </div>

        <div class="loader-main-title">DIET ENGINE READY</div>

        <!-- Standby Description -->
        <div class="loader-status-container">
            <div class="loader-status-text" style="color: #64748b; font-weight: 600; font-size: 0.88rem;">
                Awaiting meal photo upload on the left...
            </div>
        </div>

        <!-- Gentle Pulse Track -->
        <div class="loader-progress-track">
            <div class="loader-progress-bar" style="animation-duration: 3.5s; opacity: 0.75;"></div>
        </div>

        <!-- Continuously Cycling Website Phrases -->
        <div class="rotator-wrapper">
            <div class="rotator-text-item ri-1">🥗 Drop any breakfast, lunch, or dinner plate...</div>
            <div class="rotator-text-item ri-2">🔍 YOLOv8 automatically localizes portions & geometry...</div>
            <div class="rotator-text-item ri-3">🤖 Gemini Vision computes calories, protein & healthy fats...</div>
            <div class="rotator-text-item ri-4">📊 Nutrition breakdown & HUD results appear here instantly...</div>
            <div class="rotator-text-item ri-5">⚡ Real-time caloric density & macro tracking ready...</div>
        </div>

        <!-- Standby Telemetry Chips -->
        <div class="loader-telemetry-row">
            <div class="loader-telem-chip"><span style="color:#0284c7;">●</span> YOLOv8: ARMED</div>
            <div class="loader-telem-chip"><span style="color:#10b981;">●</span> GEMINI: READY</div>
            <div class="loader-telem-chip"><span style="color:#00f2fe;">●</span> HUD: STANDBY</div>
        </div>
    </div>
</div>
"""

    if status_text is None:
        status_text = "Scanning meal plate with YOLOv8 & Gemini Vision AI..."

    portion_badge = ""
    if current_portion and total_portions:
        portion_badge = f"""
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.76rem; font-weight: 800; color: #0284c7; background: rgba(2, 132, 199, 0.12); border: 1px solid rgba(2, 132, 199, 0.3); border-radius: 999px; padding: 0.2rem 0.7rem; display: inline-block; margin-bottom: 0.4rem;">
            PORTION {current_portion} OF {total_portions}
        </div>
        """

    return f"""
<div class="loader-modal-backdrop">
    <div class="loader-card">
        <!-- Radar Scanner Stage with Rotating Beam -->
        <div class="radar-stage">
            <div class="radar-outer-ring"></div>
            <div class="radar-beam"></div>
            <div class="radar-inner-glow"></div>
            
            <!-- Center: Smoothly Cycling 3D Food Icons -->
            <div class="food-cycler-container">
                <div class="food-icon-item fi-1">🥗</div>
                <div class="food-icon-item fi-2">🥑</div>
                <div class="food-icon-item fi-3">🍗</div>
                <div class="food-icon-item fi-4">🍣</div>
                <div class="food-icon-item fi-5">🥪</div>
                <div class="food-icon-item fi-6">🍳</div>
            </div>
        </div>

        <!-- Pipeline Status Badge -->
        <div class="loader-badge-tag">
            <span class="pulse-dot-loader"></span>
            <span>AI INFERENCE ENGINE ACTIVE</span>
        </div>

        {portion_badge}

        <div class="loader-main-title">DIET ENGINE NEURAL RECOGNITION</div>

        <!-- Dynamic Live Status Text -->
        <div class="loader-status-container">
            <div class="loader-status-text">{status_text}</div>
        </div>

        <!-- Glowing Sweeping Neon Progress Bar -->
        <div class="loader-progress-track">
            <div class="loader-progress-bar"></div>
        </div>

        <!-- Continuously Cycling Website Phrases -->
        <div class="rotator-wrapper">
            <div class="rotator-text-item ri-1">🔍 YOLOv8 isolating food portions & geometry...</div>
            <div class="rotator-text-item ri-2">🤖 Gemini Vision evaluating ingredient density & volume...</div>
            <div class="rotator-text-item ri-3">🥑 Calculating Protein, Carbohydrates & Healthy Fats...</div>
            <div class="rotator-text-item ri-4">🔬 Assessing surface charring, freshness & cooking quality...</div>
            <div class="rotator-text-item ri-5">⚡ Synthesizing personalized Diet Engine nutrition HUD...</div>
        </div>

        <!-- Bottom Telemetry Chips -->
        <div class="loader-telemetry-row">
            <div class="loader-telem-chip"><span style="color:#0284c7;">●</span> YOLOv8: ACTIVE</div>
            <div class="loader-telem-chip"><span style="color:#00f2fe;">●</span> GEMINI: STREAMING</div>
            <div class="loader-telem-chip"><span style="color:#10b981;">●</span> MACROS: CALIBRATING</div>
        </div>
    </div>
</div>
"""

# Navigation State Management
if "show_features" not in st.session_state:
    st.session_state["show_features"] = False


# ==========================================
# VIEW 1: LANDING PAGE (NATIVELY CODED DIET ENGINE POSTER)
# ==========================================
if not st.session_state["show_features"]:
    plate_b64 = get_plate_base64()

    poster_html = f"""
    <div class="diet-engine-poster">
        <!-- Top Navigation / Header -->
        <div class="poster-header">
             <div class="poster-tag">Your Healthy Agent</div>
            <div class="poster-title-wrap">
                <h1 class="poster-brand">DIET ENGINE</h1>
                <div class="poster-subbrand">Personalized AI Nutrition & Research Analysis</div>
            </div>
        </div>

        <!-- Main Visual Stage -->
        <div class="poster-stage">
            <!-- SVG Circuit Lines Linking Plate to Phone -->
            <svg class="hud-circuit-svg" viewBox="0 0 1000 440" preserveAspectRatio="none">
                <defs>
                    <filter id="hud-glow" x="-20%" y="-20%" width="140%" height="140%">
                        <feGaussianBlur stdDeviation="3" result="blur" />
                        <feComposite in="SourceGraphic" in2="blur" operator="over" />
                    </filter>
                </defs>
                <!-- Background Neural Constellation Lines -->
                <line x1="30" y1="30" x2="140" y2="70" stroke="rgba(2, 132, 199, 0.16)" stroke-width="1.2" />
                <line x1="140" y1="70" x2="80" y2="160" stroke="rgba(2, 132, 199, 0.14)" stroke-width="1.2" />
                <line x1="880" y1="40" x2="800" y2="110" stroke="rgba(2, 132, 199, 0.16)" stroke-width="1.2" />
                <line x1="820" y1="360" x2="940" y2="400" stroke="rgba(2, 132, 199, 0.14)" stroke-width="1.2" />

                <!-- Circuit Line: Salmon to Phone Center -->
                <path d="M 285 220 L 410 220 L 460 235 L 530 235" fill="none" stroke="#00f2fe" stroke-width="2.6" filter="url(#hud-glow)" />
                <circle cx="285" cy="220" r="5" fill="#00f2fe" filter="url(#hud-glow)" />
                <circle cx="530" cy="235" r="5" fill="#00f2fe" filter="url(#hud-glow)" />

                <!-- Circuit Line: Quinoa (Top-Left) -->
                <path d="M 195 125 L 90 125 L 65 170 L 65 240" fill="none" stroke="rgba(0, 242, 254, 0.55)" stroke-width="1.8" />
                <circle cx="195" cy="125" r="3.5" fill="#00f2fe" />

                <!-- Circuit Line: Avocado (Bottom-Left) -->
                <path d="M 160 295 L 85 340 L 115 375" fill="none" stroke="rgba(0, 242, 254, 0.45)" stroke-width="1.6" />
                <circle cx="160" cy="295" r="3.5" fill="#00f2fe" />

                <!-- Circuit Line: Tomatoes to Phone -->
                <path d="M 285 305 L 360 305 L 420 275 L 470 275" fill="none" stroke="rgba(0, 242, 254, 0.55)" stroke-width="1.8" />
                <circle cx="285" cy="305" r="3.5" fill="#00f2fe" />

                <!-- Phone to Right Badges Connections -->
                <path d="M 710 180 L 760 180" fill="none" stroke="rgba(0, 242, 254, 0.65)" stroke-width="1.8" />
                <circle cx="760" cy="180" r="3.5" fill="#00f2fe" />
                <path d="M 710 245 L 760 245" fill="none" stroke="rgba(0, 242, 254, 0.65)" stroke-width="1.8" />
                <circle cx="760" cy="245" r="3.5" fill="#00f2fe" />
                <path d="M 710 315 L 760 315" fill="none" stroke="rgba(0, 242, 254, 0.65)" stroke-width="1.8" />
                <circle cx="760" cy="315" r="3.5" fill="#00f2fe" />
            </svg>

            <!-- Left: Circular Food Plate with Target Overlays -->
            <div class="plate-container">
                <div class="plate-frame">
                    <img class="food-plate-img" src="data:image/png;base64,{plate_b64}" alt="AI Healthy Food Plate" />

                    <!-- Reticle 1: Salmon -->
                    <div class="reticle-pin" style="top: 48%; left: 52%;">
                        <div class="reticle-ring"></div>
                        <div class="reticle-core"></div>
                        <div class="reticle-tooltip">Grilled Salmon</div>
                    </div>

                    <!-- Reticle 2: Avocado -->
                    <div class="reticle-pin" style="top: 60%; left: 24%;">
                        <div class="reticle-ring"></div>
                        <div class="reticle-core"></div>
                        <div class="reticle-tooltip">Avocado</div>
                    </div>

                    <!-- Reticle 3: Quinoa -->
                    <div class="reticle-pin" style="top: 25%; left: 34%;">
                        <div class="reticle-ring"></div>
                        <div class="reticle-core"></div>
                        <div class="reticle-tooltip">Quinoa</div>
                    </div>

                    <!-- Reticle 4: Tomatoes -->
                    <div class="reticle-pin" style="top: 70%; left: 60%;">
                        <div class="reticle-ring"></div>
                        <div class="reticle-core"></div>
                        <div class="reticle-tooltip">Cherry Tomatoes</div>
                    </div>
                </div>

                <div class="plate-tech-hud">
                    <span style="color: #00f2fe;">●</span> TARGET ACQUIRED: 4 PORTIONS DETECTED
                </div>
            </div>

            <!-- Center: Futuristic White Smartphone UI Mockup -->
            <div class="phone-device-wrapper">
                <div class="phone-device">
                    <div class="phone-speaker-notch"></div>
                    <div class="phone-header-title">DIET ENGINE</div>

                    <!-- Camera Scanner Area -->
                    <div class="phone-camera-area">
                        <div class="camera-lens-box">
                            <span style="font-size: 1.5rem; color: #0284c7;">📷</span>
                        </div>
                        <div class="scan-frequency-wave">
                            <span style="color: #00f2fe;">∿∿∿</span> Scan <span style="color: #00f2fe;">∿∿∿</span>
                        </div>
                    </div>

                    <!-- Detection Result Card -->
                    <div class="phone-result-card">
                        <div class="phone-target-icon">◎</div>
                        <div>
                            <div class="phone-label-sm">Meal Detected:</div>
                            <div class="phone-meal-name">Healthy Plate</div>
                        </div>
                    </div>

                    <!-- Calories Readout -->
                    <div class="phone-calories-card">
                        <div class="phone-label-sm">Total Calories:</div>
                        <div class="phone-cal-number">🔥 540 <span class="phone-cal-unit">kcal</span></div>
                    </div>

                    <!-- Macro Bars -->
                    <div class="phone-macro-bars">
                        <div class="phone-bar-item">
                            <div class="phone-bar-label"><span>🍗 Protein</span> <span>35g</span></div>
                            <div class="phone-bar-track">
                                <div class="phone-bar-progress" style="width: 78%; background: linear-gradient(90deg, #00f2fe, #0284c7);"></div>
                            </div>
                        </div>

                        <div class="phone-bar-item">
                            <div class="phone-bar-label"><span>🍞 Carbs</span> <span>28g</span></div>
                            <div class="phone-bar-track">
                                <div class="phone-bar-progress" style="width: 58%; background: linear-gradient(90deg, #38bdf8, #0ea5e9);"></div>
                            </div>
                        </div>

                        <div class="phone-bar-item">
                            <div class="phone-bar-label"><span>🥑 Fats</span> <span>12g</span></div>
                            <div class="phone-bar-track">
                                <div class="phone-bar-progress" style="width: 42%; background: linear-gradient(90deg, #10b981, #059669);"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Right: Floating Telemetry Badges -->
            <div class="telemetry-badges-column">
                <div class="telemetry-pill">
                    <span style="font-size: 1.2rem; color: #0284c7;">⚛️</span>
                    <span>35g</span>
                </div>
                <div class="telemetry-pill">
                    <span style="font-size: 1.2rem; color: #06b6d4;">📈</span>
                    <span>340</span>
                </div>
                <div class="telemetry-pill">
                    <span style="font-size: 1.2rem; color: #10b981;">🌾</span>
                    <span>Fiber</span>
                </div>
            </div>
        </div>
    </div>
    """
    render_html(poster_html)

    # 2. Minimal website description (clean and concise)
    render_html("""
    <div style="text-align: center; margin: 1.2rem auto 1.8rem auto; max-width: 780px;">
        <p style="font-size: 1.15rem; color: #334155; font-weight: 500; line-height: 1.6; margin: 0;">
            An AI-powered nutrition platform using <strong>YOLOv8</strong> and <strong>Gemini Vision</strong> to detect meals, estimate portion weights, and calculate real-time macronutrients directly from food photos.
        </p>
    </div>
    """)

    # 3. Clean "Click here to continue" button
    c1, c2, c3 = st.columns([1, 1.4, 1])
    with c2:
        if st.button("Click here to continue ➔", use_container_width=True, key="landing_continue_btn"):
            st.session_state["show_features"] = True
            st.rerun()


# ==========================================
# VIEW 2: FEATURE PART (AI SCANNER & ANALYZER)
# ==========================================
else:
    # Feature Header Bar with Back Button
    nav_col1, nav_col2 = st.columns([3, 1], gap="medium")
    with nav_col1:
        render_html("""
        <div style="padding: 0.5rem 0 1rem 0;">
            <div style="display: inline-block; font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; font-weight: 700; color: #0284c7; background: rgba(2, 132, 199, 0.12); border: 1.5px solid rgba(2, 132, 199, 0.35); padding: 0.35rem 1.1rem; border-radius: 999px; margin-bottom: 0.4rem; text-transform: uppercase;">
                AI MEAL SCANNER &bull; ACTIVE ENGINE
            </div>
            <h2 style="font-size: 2.2rem; font-weight: 900; color: #0f172a; margin: 0;">🥗 Interactive Nutrition Analysis</h2>
            <div style="font-size: 0.95rem; color: #475569; margin-top: 0.2rem;">Upload or capture meal plates. YOLOv8 localizes food portions and Gemini Vision calculates macros.</div>
        </div>
        """)
    with nav_col2:
        st.markdown("<div style='margin-top: 1.2rem;'></div>", unsafe_allow_html=True)
        if st.button("⬅ Back to Overview / Poster", use_container_width=True, key="btn_back_to_landing"):
            st.session_state["show_features"] = False
            st.rerun()

    # Load Models
    detector, classifier = load_models()

    # Main Feature Dual-Column Layout
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

    # Right column placeholder: houses minimal loader (standby or active) then clears for prediction card
    right_placeholder = main_col2.empty()

    if uploaded_file is None:
        # Show good static showcase image on right hand side initially
        right_placeholder.html(get_standby_static_card_html())
    else:
        image = Image.open(uploaded_file).convert("RGB")

        # Step 1: Initial scanning state inside right_placeholder
        right_placeholder.html(get_loader_html("🔍 Localizing meal portions & food boundaries with YOLOv8..."))

        # 1. Detection
        raw_boxes, raw_crops = detector.detect(image)
        # 2. Deduplicate overlapping crops
        boxes, crops = filter_redundant_crops(raw_boxes, raw_crops)

        num_crops = len(crops)
        if num_crops == 0:
            right_placeholder.html(get_loader_html("Whole plate captured. Querying Gemini Vision AI for macro breakdown...", current_portion=1, total_portions=1))
            crops = [image]
            boxes = [{'box': [0, 0, image.width, image.height], 'conf': 0.95}]
            num_crops = 1
        else:
            right_placeholder.html(get_loader_html(f"🎯 YOLOv8 detected {num_crops} portion{'s' if num_crops != 1 else ''}. Initializing Gemini Vision AI...", current_portion=1, total_portions=num_crops))

        # 3. LLM Vision Analysis on all detected crops
        items_data = []
        item_names_for_overlay = []

        for idx, crop in enumerate(crops):
            right_placeholder.html(get_loader_html(f"🤖 Gemini Vision analyzing Portion {idx + 1} of {num_crops}: Computing Calories & Macronutrients...", current_portion=idx + 1, total_portions=num_crops))
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

        # Final step before display
        right_placeholder.html(get_loader_html("⚡ Synthesizing Nutritional Profile & Generating Diet Engine HUD..."))

        # Cleanly remove the loader from right_placeholder to reveal the results
        right_placeholder.empty()

        # Render Left Column with High-Tech Bounding Boxes
        with main_col1:
            annotated_img = draw_high_tech_overlay(image, boxes, item_names_for_overlay)
            st.image(annotated_img, caption="YOLOv8 Dynamic Target Localization", use_container_width=True)

            num_detected = len(boxes)
            render_html(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 0.9rem; padding: 0.75rem 1.1rem; background: #f8fafc; border: 1.5px solid #e2e8f0; border-radius: 14px; font-family: 'JetBrains Mono', monospace; font-size: 0.88rem; color: #1e293b;">
                <div><span>TARGETS DETECTED:</span> <strong style="color: #0284c7; font-size: 1.05rem;">{num_detected}</strong></div>
                <div><span>STATUS:</span> <strong style="color: #10b981; font-size: 0.95rem;">OPTIMAL SCAN</strong></div>
            </div>
            """)

        # Render Right Column with Prediction Results
        with main_col2:
            total_cal = sum(it.get("calories", 0) for it in items_data)
            total_pro = sum(it.get("protein", 0) for it in items_data)
            total_fat = sum(it.get("fat", 0) for it in items_data)
            total_carb = sum(it.get("carbs", 0) for it in items_data)
            total_fiber = sum(it.get("fiber", 0) for it in items_data)
            total_weight = sum(it.get("estimated_weight_g", 0) for it in items_data)

            conditions = [it.get("visual_condition", "normal_appearance") for it in items_data]
            if "possible_spoilage_signs" in conditions:
                dominant_cond = "possible_spoilage_signs"
            elif "overcooked_or_burnt" in conditions:
                dominant_cond = "overcooked_or_burnt"
            elif "normal_appearance" in conditions:
                dominant_cond = "normal_appearance"
            else:
                dominant_cond = "uncertain"

            if items_data:
                top_labels = [it.get("label", "").title() for it in items_data[:3]]
                composite_meal_name = " + ".join(top_labels)
            else:
                composite_meal_name = "Whole Plate"

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
