"""Interactive Gradio application for the FoodMacros pipeline.

Complete high-end redesign:
- Clean camera-first experience with 1-click sample meal previews
- High-visibility YOLO bounding boxes with identified food tags
- Radial SVG Calorie Donut gauge and macro distribution split bars
- Instant Light Mode / Dark Mode toggle
- Separate Single Dish Scanner and Multi-Dish Plate Scanner
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from foodmacros.config import AppConfig, PROJECT_ROOT, load_config  # noqa: E402
from foodmacros.detection.visualize import draw_detections  # noqa: E402
from foodmacros.env_checks import has_package  # noqa: E402
from foodmacros.multi_pipeline import MultiDishPipeline  # noqa: E402
from foodmacros.multi_reporting import (  # noqa: E402
    format_multi_analysis,
    multi_analysis_to_dict,
)
from foodmacros.pipeline import FoodAnalysisPipeline  # noqa: E402
from foodmacros.reporting import (  # noqa: E402
    LIMITATIONS,
    analysis_to_dict,
    format_nutrients,
)


SPLASH_3D_HTML = ""

BRAND_HTML = """
<div class="fm-brand-container">
  <div class="fm-brand">
    <div class="fm-logo" aria-hidden="true">
      <svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor"
           stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
      </svg>
    </div>
    <div>
      <div class="fm-title">FoodMacros <span class="fm-title-gradient">Vision AI</span></div>
      <div class="fm-subtitle">Autonomous Food Detection · Mass Estimation · Macronutrients</div>
    </div>
  </div>
  <div class="fm-header-right">
    <div class="fm-status-pill">
      <span class="fm-pulse-dot"></span>
      <span>YOLOv8 Engine Active</span>
    </div>
  </div>
</div>
"""

CSS = """
:root {
  --layout-max-width: 98% !important;
  --fm-green-950: #0b2e1c;
  --fm-green-900: #14532d;
  --fm-green-800: #166534;
  --fm-green-700: #15803d;
  --fm-green-600: #16a34a;
  --fm-green-500: #22c55e;
  --fm-green-100: #dcfce7;
  --fm-green-50: #f0fdf4;
  
  --fm-cyan-700: #0369a1;
  --fm-cyan-500: #0284c7;
  --fm-cyan-50: #f0f9ff;
  
  --fm-amber-700: #b45309;
  --fm-amber-500: #d97706;
  --fm-amber-50: #fffbeb;
  
  --fm-rose-700: #be123c;
  --fm-rose-500: #e11d48;
  --fm-rose-50: #fff1f2;
  
  --fm-purple-700: #6d28d9;
  --fm-purple-500: #7c3aed;
  --fm-purple-50: #faf5ff;
  
  /* Pure Aesthetic Light Mode Theme */
  --fm-bg: #f8fafc;
  --fm-canvas: #f1f5f9;
  --fm-card-bg: #ffffff;
  --fm-card-sub: #f8fafc;
  --fm-border: #e2e8f0;
  --fm-border-focus: #16a34a;
  --fm-ink: #0f172a;
  --fm-body: #1e293b;
  --fm-muted: #475569;
  --fm-subtext: #64748b;
  --fm-stat-bg: #f8fafc;
  --fm-stat-border: #e2e8f0;
}

/* Base resets & typography */
html, body, .gradio-container {
  background-color: #f8fafc !important;
  color: #1e293b !important;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
  font-size: 14.5px;
  line-height: 1.6;
}

.gradio-container,
.gradio-container.gradio-container-5-0-0,
div.gradio-container,
.gradio-container .main,
.gradio-container > .wrap,
.gradio-container .contain {
  max-width: 98% !important;
  width: 98% !important;
  margin: 0 auto !important;
  padding: 12px 18px !important;
}

.gradio-container button,
.gradio-container input,
.gradio-container textarea,
.gradio-container select,
.gradio-container table {
  font-family: inherit !important;
}

/* Header & Brand Container */
.fm-brand-container {
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
  flex-wrap: wrap;
  gap: 14px;
}

.fm-header-row {
  background: #ffffff !important;
  border: 1px solid #e2e8f0 !important;
  border-radius: 20px !important;
  padding: 16px 28px !important;
  margin-bottom: 20px !important;
  align-items: center !important;
  box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.05) !important;
}

.fm-brand {
  display: flex;
  align-items: center;
  gap: 14px;
}

.fm-logo {
  width: 48px;
  height: 48px;
  border-radius: 14px;
  background: linear-gradient(145deg, #14532d, #16a34a);
  border: 1px solid #86efac;
  color: #ffffff;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 6px 16px rgba(20, 83, 45, 0.2);
}

.fm-title {
  font-size: 22px;
  font-weight: 800;
  letter-spacing: -0.03em;
  color: #0f172a;
  line-height: 1.2;
}

.fm-title-gradient {
  background: linear-gradient(90deg, #15803d 0%, #0d9488 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.fm-subtitle {
  font-size: 13px;
  color: #64748b;
  font-weight: 500;
  margin-top: 2px;
}

.fm-header-right {
  display: flex;
  align-items: center;
  gap: 12px;
}

.fm-status-pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background: #ecfdf5;
  border: 1px solid #a7f3d0;
  padding: 6px 14px;
  border-radius: 9999px;
  font-size: 12px;
  font-weight: 700;
  color: #065f46;
}

.fm-pulse-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #10b981;
  box-shadow: 0 0 8px #10b981;
  animation: fm-pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
}

@keyframes fm-pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.35; transform: scale(0.8); }
}

/* Tabs Navigation */
.fm-main-tabs > .tab-nav {
  background: #ffffff !important;
  border: 1px solid #e2e8f0 !important;
  border-radius: 16px !important;
  padding: 6px !important;
  gap: 8px !important;
  margin-bottom: 22px !important;
  box-shadow: 0 2px 10px rgba(15, 23, 42, 0.03) !important;
}

.fm-main-tabs > .tab-nav > button {
  border-radius: 12px !important;
  font-size: 14px !important;
  font-weight: 700 !important;
  padding: 10px 22px !important;
  color: #64748b !important;
  border: 1px solid transparent !important;
  transition: all 0.2s ease !important;
  background: transparent !important;
}

.fm-main-tabs > .tab-nav > button:hover {
  color: #0f172a !important;
  background: #f8fafc !important;
}

.fm-main-tabs > .tab-nav > button.selected {
  background: #f0fdf4 !important;
  border: 1px solid #86efac !important;
  color: #166534 !important;
  font-weight: 800 !important;
  box-shadow: 0 2px 8px rgba(22, 101, 52, 0.08) !important;
}

/* Bento Elevated Cards */
.fm-card {
  background: #ffffff !important;
  border: 1px solid #e2e8f0 !important;
  border-radius: 20px !important;
  padding: 22px !important;
  box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.05) !important;
  transition: border-color 0.2s ease, transform 0.2s ease !important;
}

.fm-summary {
  background: linear-gradient(180deg, #ffffff 0%, #f0fdf4 100%) !important;
  border: 1px solid #bbf7d0 !important;
}

.fm-section-title h2 {
  font-size: 18px;
  font-weight: 800;
  letter-spacing: -0.02em;
  color: #0f172a;
  margin: 0;
}

.fm-section-title p {
  color: #64748b;
  font-size: 13px;
  margin: 4px 0 0;
}

.fm-summary-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 12px;
}

.fm-summary-head h3 {
  margin: 0;
  font-size: 17px;
  font-weight: 800;
  color: #0f172a;
  letter-spacing: -0.02em;
}

.fm-bento-hero {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 14px 16px;
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 18px;
  transition: all 0.2s ease;
}

.fm-card .fm-bento-hero {
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
  box-shadow: none !important;
  border-radius: 0 !important;
}

.fm-weight-card {
  padding: 14px 20px !important;
  margin-top: 12px !important;
  margin-bottom: 12px !important;
}

.fm-nutrition-card {
  padding: 18px 22px !important;
}

.fm-bento-title {
  font-size: 15px;
  font-weight: 800;
  color: #0f172a;
  letter-spacing: -0.01em;
  display: flex;
  align-items: center;
  gap: 6px;
}

.fm-bento-standby-row {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 0;
  font-size: 13px;
  color: #64748b;
}

/* Badges & Pills */
.fm-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 12px;
  border-radius: 9999px;
  background: #f0fdf4;
  border: 1px solid #bbf7d0;
  color: #15803d;
  font-size: 12px;
  font-weight: 700;
  white-space: nowrap;
}

.fm-badge-muted {
  background: #f1f5f9;
  border-color: #cbd5e1;
  color: #64748b;
}

.fm-badge-cyan {
  background: #f0f9ff;
  border-color: #bae6fd;
  color: #0284c7;
}

.fm-badge-green {
  background: #f0fdf4;
  border-color: #86efac;
  color: #15803d;
}

.fm-badge-amber {
  background: #fffbeb;
  border-color: #fde68a;
  color: #b45309;
}

.fm-badge-purple {
  background: #faf5ff;
  border-color: #ddd6fe;
  color: #7c3aed;
}

/* Top-k Chips */
.fm-topk-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 8px;
}

.fm-topk-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  padding: 4px 10px;
  border-radius: 8px;
  font-size: 12px;
}

.fm-topk-chip .name {
  color: #334155;
  font-weight: 600;
}

.fm-topk-chip .prob {
  color: #0284c7;
  font-weight: 800;
  font-family: monospace;
}

/* 4-Cube Bento Macro Mini-divs */
.fm-macro-4grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 10px;
  margin: 10px 0;
}

.fm-cube {
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  padding: 10px 6px;
  border-radius: 14px;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.fm-cube:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(15, 23, 42, 0.05);
}

.fm-cube .label {
  font-size: 10px;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-bottom: 2px;
}

.fm-cube .val {
  font-size: 17px;
  font-weight: 900;
  line-height: 1.1;
}

.fm-cube .sub {
  font-size: 10px;
  font-weight: 600;
  opacity: 0.85;
}

.fm-cube-cals {
  background: #f0fdf4;
  border: 1px solid #bbf7d0;
}
.fm-cube-cals .label, .fm-cube-cals .val, .fm-cube-cals .sub { color: #15803d; }

.fm-cube-protein {
  background: #f0f9ff;
  border: 1px solid #bae6fd;
}
.fm-cube-protein .label, .fm-cube-protein .val, .fm-cube-protein .sub { color: #0284c7; }

.fm-cube-carbs {
  background: #fffbeb;
  border: 1px solid #fde68a;
}
.fm-cube-carbs .label, .fm-cube-carbs .val, .fm-cube-carbs .sub { color: #b45309; }

.fm-cube-fats {
  background: #fff1f2;
  border: 1px solid #fecdd3;
}
.fm-cube-fats .label, .fm-cube-fats .val, .fm-cube-fats .sub { color: #be123c; }

/* Micro Nutrient Chips */
.fm-micro-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 8px;
}

.fm-micro-chip {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: 11.5px;
  color: #475569;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  padding: 3px 10px;
  border-radius: 9999px;
}

.fm-micro-chip strong {
  color: #0f172a;
}

/* Weight Strip */
.fm-weight-strip {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  width: 100%;
}

.fm-weight-left {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.fm-weight-right {
  display: flex;
  align-items: center;
  gap: 12px;
}

.fm-weight-val {
  font-size: 22px;
  font-weight: 900;
  color: #0f172a;
}

.fm-weight-unit {
  font-size: 13px;
  font-weight: 700;
  color: #15803d;
}

.fm-weight-basis-pill {
  font-size: 11.5px;
  font-weight: 700;
  padding: 4px 10px;
  border-radius: 9999px;
  background: #f1f5f9;
  border: 1px solid #cbd5e1;
  color: #475569;
}

/* Macro Caloric Ratio Bar */
.fm-macro-bar-container {
  margin: 10px 0 6px;
}

.fm-macro-bar-title {
  display: flex;
  justify-content: space-between;
  font-size: 11.5px;
  font-weight: 700;
  color: #64748b;
  margin-bottom: 6px;
}

.fm-ratio-legend {
  display: flex;
  align-items: center;
  gap: 12px;
  white-space: nowrap !important;
}

.fm-ratio-legend span {
  white-space: nowrap !important;
  font-weight: 800;
  font-size: 11.5px;
}

.fm-macro-bar {
  display: flex;
  height: 8px;
  border-radius: 9999px;
  overflow: hidden;
  background: #e2e8f0;
}

.fm-macro-bar-protein { background: #0284c7; }
.fm-macro-bar-carbs { background: #d97706; }
.fm-macro-bar-fats { background: #e11d48; }

/* Multi-Dish Bento Grid & Cards */
.fm-dish-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(270px, 1fr));
  gap: 14px;
  margin-top: 14px;
}

.fm-dish-card {
  background: #ffffff !important;
  border: 1px solid #e2e8f0 !important;
  border-radius: 18px !important;
  padding: 14px 16px !important;
  box-shadow: 0 4px 16px -2px rgba(15, 23, 42, 0.04) !important;
  transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease !important;
  display: flex;
  flex-direction: column;
  gap: 6px;
  position: relative;
  overflow: hidden;
}

.fm-dish-card:hover {
  transform: translateY(-2px) !important;
  border-color: #86efac !important;
  box-shadow: 0 8px 24px -4px rgba(15, 23, 42, 0.08) !important;
}

.fm-dish-num-badge {
  width: 28px;
  height: 28px;
  border-radius: 8px;
  font-size: 12px;
  font-weight: 800;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.fm-dish-macro-chips {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 6px;
  margin-top: 8px;
}

.fm-dish-macro-chips span {
  font-size: 11px;
  font-weight: 800;
  padding: 4px 8px;
  border-radius: 8px;
  text-align: center;
}

.chip-cal {
  background: #f0fdf4 !important;
  border: 1px solid #bbf7d0 !important;
  color: #15803d !important;
}

.chip-p {
  background: #f0f9ff !important;
  border: 1px solid #bae6fd !important;
  color: #0284c7 !important;
}

.chip-c {
  background: #fffbeb !important;
  border: 1px solid #fde68a !important;
  color: #b45309 !important;
}

.chip-f {
  background: #fff1f2 !important;
  border: 1px solid #fecdd3 !important;
  color: #be123c !important;
}

/* Image Viewports & Dropzones: Clean Light, No Dark Void */
.gradio-container .image-container,
.gradio-container .empty,
.gradio-container .upload-button,
.gradio-container [data-testid="image"],
.gradio-container .image-frame,
.gradio-container .viewport,
.fm-image,
.fm-image .image-container,
.fm-image img {
  background: #f8fafc !important;
  border: 1px solid #e2e8f0 !important;
  border-radius: 16px !important;
  color: #64748b !important;
}

.gradio-container .upload-button:hover {
  border-color: #16a34a !important;
  background: #f0fdf4 !important;
}

/* Buttons */
.fm-primary {
  background: linear-gradient(135deg, #15803d 0%, #16a34a 100%) !important;
  border: 1px solid #166534 !important;
  color: #ffffff !important;
  font-weight: 800 !important;
  border-radius: 12px !important;
  padding: 12px 24px !important;
  box-shadow: 0 4px 14px rgba(22, 163, 74, 0.25) !important;
  transition: all 0.2s ease !important;
  cursor: pointer !important;
}

.fm-primary:hover {
  background: linear-gradient(135deg, #166534 0%, #15803d 100%) !important;
  box-shadow: 0 6px 20px rgba(22, 163, 74, 0.35) !important;
  transform: translateY(-1px) !important;
}

.fm-scan-btn {
  font-size: 15px !important;
  padding: 13px 26px !important;
  letter-spacing: 0.02em !important;
  margin-top: 6px !important;
  width: 100% !important;
}

/* Sliders, Inputs and Form Controls */
.gradio-container .gradio-slider,
.gradio-container .gradio-number {
  background: transparent !important;
  border: none !important;
}

.gradio-container label span,
.gradio-container .block label span,
.gradio-container .block .block-info {
  background: transparent !important;
  color: #475569 !important;
  font-size: 11.5px !important;
  font-weight: 700 !important;
  letter-spacing: 0.04em !important;
  text-transform: uppercase !important;
  padding: 0 0 4px 0 !important;
  border: none !important;
  box-shadow: none !important;
}

.gradio-container input[type="number"],
.gradio-container input[type="text"] {
  background: #ffffff !important;
  border: 1px solid #cbd5e1 !important;
  border-radius: 10px !important;
  color: #0f172a !important;
  font-weight: 700 !important;
  font-size: 13px !important;
}

.gradio-container input[type="number"]:focus,
.gradio-container input[type="text"]:focus {
  border-color: #16a34a !important;
  outline: 2px solid rgba(22, 163, 74, 0.2) !important;
}

.gradio-container input[type="range"] {
  accent-color: #16a34a !important;
}

/* 1-Click Samples table */
.gradio-container .gr-examples {
  background: #ffffff !important;
  border: 1px solid #e2e8f0 !important;
  border-radius: 16px !important;
  padding: 12px 16px !important;
  margin-top: 10px !important;
  max-width: 100% !important;
  overflow-x: auto !important;
  box-sizing: border-box !important;
  box-shadow: 0 2px 12px rgba(15, 23, 42, 0.03) !important;
}

.gradio-container .gr-examples table {
  width: 100% !important;
  border-collapse: collapse !important;
  font-size: 12.5px !important;
}

.gradio-container .gr-examples tr {
  border-bottom: 1px solid #f1f5f9 !important;
  transition: background 0.15s ease !important;
}

.gradio-container .gr-examples tr:hover {
  background: #f0fdf4 !important;
  cursor: pointer !important;
}

.gradio-container .gr-examples img {
  border-radius: 8px !important;
  max-height: 48px !important;
  width: auto !important;
  object-fit: cover !important;
}

/* About & Limitations */
.fm-about-copy .prose {
  color: #1e293b !important;
  font-size: 14.5px !important;
  line-height: 1.65 !important;
}

.fm-about-copy .prose h3 {
  color: #14532d !important;
  font-size: 18px !important;
  font-weight: 800 !important;
  margin: 18px 0 8px !important;
}

.fm-about-copy .prose strong {
  color: #0f172a !important;
}

.fm-limitations .prose {
  color: #64748b !important;
  font-size: 13px !important;
  line-height: 1.6 !important;
}

/* Layout Rows & Responsive Flow */
.fm-layout-row {
  display: flex !important;
  flex-direction: row !important;
  flex-wrap: nowrap !important;
  gap: 20px !important;
  align-items: stretch !important;
  width: 100% !important;
  box-sizing: border-box !important;
  margin-bottom: 16px !important;
}

.fm-layout-row > * {
  min-width: 0 !important;
  box-sizing: border-box !important;
}

.fm-layout-row > *:first-child {
  flex: 1 1 48% !important;
  max-width: 50% !important;
}

.fm-layout-row > *:last-child {
  flex: 1 1 52% !important;
  max-width: 52% !important;
}

.fm-samples-fullrow {
  width: 100% !important;
  margin-top: 16px !important;
  margin-bottom: 16px !important;
}

.fm-samples-fullrow .fm-card {
  padding: 14px 20px !important;
}

.fm-compact-controls {
  display: flex !important;
  gap: 14px !important;
  margin: 10px 0 !important;
  align-items: center !important;
}

.fm-compact-controls > * {
  flex: 1 1 0 !important;
  min-width: 0 !important;
}

.fm-note {
  color: #64748b;
  font-size: 13px;
  line-height: 1.55;
}

.fm-pred-name {
  font-size: 26px;
  line-height: 1.15;
  font-weight: 800;
  color: #0f172a;
  letter-spacing: -0.03em;
  text-transform: capitalize;
}

@media (max-width: 900px) {
  .gradio-container {
    padding: 12px !important;
  }

  .fm-layout-row {
    flex-direction: column !important;
    flex-wrap: wrap !important;
  }

  .fm-layout-row > *:first-child,
  .fm-layout-row > *:last-child,
  .fm-layout-row > * {
    max-width: 100% !important;
    width: 100% !important;
  }
}
"""


def create_pipeline(config: AppConfig, model_dir=None, regressor_dir=None):
    pipeline, resolved = FoodAnalysisPipeline.from_discovery(
        config, model_dir=model_dir, mass_regressor_dir=regressor_dir
    )
    return pipeline, resolved


def create_multi_pipeline(config: AppConfig, model_dir=None):
    """Build the multi-dish pipeline from the project's default artifacts."""
    return MultiDishPipeline.from_defaults(config, model_dir=model_dir)


def _classification_text(result) -> str:
    prediction = result.prediction
    if prediction is None or not prediction.available:
        reason = prediction.reason if prediction else "not run"
        return f"Unavailable: {reason}"
    lines = [f"{prediction.label}  ({prediction.confidence:.1%})"]
    for name, prob in prediction.top_k:
        lines.append(f"  {name}: {prob:.1%}")
    return "\n".join(lines)


def _macros_text(result) -> str:
    if not result.nutrition.available:
        return f"Unavailable: {result.nutrition.reason}"
    header = f"{result.nutrition.label} — {result.nutrition.basis} ({result.nutrition.grams:g} g)"
    return header + "\n" + format_nutrients(result.nutrition.nutrients)


def _clean_measured(measured):
    if measured is None or str(measured).strip() == "":
        return None
    try:
        candidate = float(measured)
    except (TypeError, ValueError):
        return None
    return candidate if candidate > 0 else None


_NUTRITION_META = [
    ("calories", "Calories", "kcal", 0, "flame"),
    ("protein", "Protein", "g", 1, "protein"),
    ("carbohydrates", "Carbohydrates", "g", 1, "carbs"),
    ("fats", "Fat", "g", 1, "fat"),
    ("fiber", "Fiber", "g", 1, "fiber"),
    ("sugars", "Sugars", "g", 1, "sugars"),
    ("sodium", "Sodium", "mg", 0, "sodium"),
]


def _fmt_value(value, digits: int = 0, suffix: str = "") -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    return f"{number:,.{digits}f}{suffix}"


def _icon(name: str) -> str:
    paths = {
        "dishes": '<path d="M4 7h16M6 7l1 13h10l1-13M9 7V4h6v3"/>',
        "weight": '<path d="M12 3a4 4 0 0 0-3.8 5.2L5 21h14L15.8 8.2A4 4 0 0 0 12 3Z"/>',
        "flame": '<path d="M12 3c3 4 5 6 5 10a5 5 0 0 1-10 0c0-2 1-4 3-6 0 2 1 3 2 3 1-2 0-4 0-7Z"/>',
        "protein": '<path d="M4 4h16v16H4zM8 8h8v8H8z"/>',
        "carbs": '<path d="M5 12c0-4 3-7 7-7s7 3 7 7-3 7-7 7-7-3-7-7ZM5 12h14"/>',
        "fat": '<path d="M12 3c4 3 6 6 6 10a6 6 0 0 1-12 0c0-4 2-7 6-10Z"/>',
        "fiber": '<path d="M12 3v18M12 7c-3 0-5-2-5-5 3 0 5 2 5 5ZM12 13c3 0 5-2 5-5-3 0-5 2-5 5ZM12 18c-3 0-5-2-5-5 3 0 5 2 5 5Z"/>',
        "sugars": '<path d="M4 4h16v16H4zM8 9h8M8 14h5"/>',
        "sodium": '<path d="M12 3v18M8 7h8M8 17h8"/>',
    }
    body = paths.get(name, "")
    return (
        '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" '
        'stroke="currentColor" stroke-width="1.8" stroke-linecap="round" '
        f'stroke-linejoin="round" style="vertical-align:-2px;margin-right:6px;">{body}</svg>'
    )


def _stat_html(label: str, value: str, icon: str) -> str:
    return (
        '<div class="fm-stat">'
        f'<div class="fm-stat-label">{_icon(icon)}{html.escape(label)}</div>'
        f'<div class="fm-stat-value">{value}</div>'
        "</div>"
    )


def _section_html(title: str, description: str = "") -> str:
    note = f'<p class="fm-note">{html.escape(description)}</p>' if description else ""
    return (
        '<div class="fm-section-title">'
        f"<h2>{html.escape(title)}</h2>{note}"
        "</div>"
    )


def _macro_donut_svg(calories: float | None, protein: float | None, carbs: float | None, fat: float | None) -> str:
    p = float(protein or 0.0) * 4.0
    c = float(carbs or 0.0) * 4.0
    f = float(fat or 0.0) * 9.0
    total_cal = float(calories or (p + c + f) or 0.0)
    
    r = 52
    circumference = 2 * math.pi * r
    
    if total_cal <= 0 or (p + c + f) <= 0:
        return (
            '<svg viewBox="0 0 130 130" width="105" height="105">'
            f'<circle cx="65" cy="65" r="{r}" fill="none" stroke="#e2e8f0" stroke-width="11"/>'
            '<text x="65" y="62" text-anchor="middle" fill="#0f172a" font-size="19" font-weight="800">0</text>'
            '<text x="65" y="78" text-anchor="middle" fill="#64748b" font-size="10" font-weight="700">kcal</text>'
            '</svg>'
        )

    sum_cals = max(p + c + f, 1.0)
    p_len = (p / sum_cals) * circumference
    c_len = (c / sum_cals) * circumference
    f_len = (f / sum_cals) * circumference

    offset_p = 0.0
    offset_c = -p_len
    offset_f = -(p_len + c_len)

    svg = f"""
    <svg viewBox="0 0 130 130" width="105" height="105" style="transform: rotate(-90deg);">
      <circle cx="65" cy="65" r="{r}" fill="none" stroke="#e2e8f0" stroke-width="11"/>
      <circle cx="65" cy="65" r="{r}" fill="none" stroke="#0284c7" stroke-width="11"
              stroke-dasharray="{p_len:.2f} {circumference:.2f}" stroke-dashoffset="{offset_p:.2f}" stroke-linecap="round"/>
      <circle cx="65" cy="65" r="{r}" fill="none" stroke="#d97706" stroke-width="11"
              stroke-dasharray="{c_len:.2f} {circumference:.2f}" stroke-dashoffset="{offset_c:.2f}"/>
      <circle cx="65" cy="65" r="{r}" fill="none" stroke="#e11d48" stroke-width="11"
              stroke-dasharray="{f_len:.2f} {circumference:.2f}" stroke-dashoffset="{offset_f:.2f}"/>
    </svg>
    <div style="position: absolute; text-align: center; pointer-events: none;">
      <div style="font-size: 20px; font-weight: 800; color: #0f172a; line-height: 1;">{int(round(total_cal))}</div>
      <div style="font-size: 10px; font-weight: 700; color: #64748b; text-transform: uppercase; margin-top: 2px;">kcal</div>
    </div>
    """
    return f'<div style="position: relative; display: flex; align-items: center; justify-content: center; width: 105px; height: 105px;">{svg}</div>'


def _macro_bar_html(protein: float | None, carbs: float | None, fat: float | None) -> str:
    p = float(protein or 0.0) * 4.0
    c = float(carbs or 0.0) * 4.0
    f = float(fat or 0.0) * 9.0
    total = p + c + f
    if total <= 0:
        return ""
    p_pct = (p / total) * 100.0
    c_pct = (c / total) * 100.0
    f_pct = (f / total) * 100.0
    return f"""
    <div class="fm-macro-bar-container">
      <div class="fm-macro-bar-title">
        <span>Macro Caloric Ratio</span>
        <span class="fm-ratio-legend">
          <span style="color:#0284c7;white-space:nowrap;">● P {p_pct:.0f}%</span>
          <span style="color:#d97706;white-space:nowrap;">● C {c_pct:.0f}%</span>
          <span style="color:#e11d48;white-space:nowrap;">● F {f_pct:.0f}%</span>
        </span>
      </div>
      <div class="fm-macro-bar">
        <div class="fm-macro-bar-protein" style="width: {p_pct:.1f}%;"></div>
        <div class="fm-macro-bar-carbs" style="width: {c_pct:.1f}%;"></div>
        <div class="fm-macro-bar-fats" style="width: {f_pct:.1f}%;"></div>
      </div>
    </div>
    """


def _plate_summary_html(result, reason=None) -> str:
    if result is None:
        note = html.escape(reason or "YOLOv8 ready · Scan a plate to localize dishes and calculate macros.")
        return (
            '<div class="fm-bento-hero">'
            '<div class="fm-summary-head"><div class="fm-bento-title">🍱 Plate Summary</div><span class="fm-badge fm-badge-muted">Standby</span></div>'
            '<div class="fm-bento-standby-row">'
            '<span class="fm-pulse-dot"></span>'
            f'<span class="fm-subtle">{note}</span>'
            '</div>'
            '</div>'
        )

    dishes = len(result.dishes)
    included = list(result.included) if result.included else [True] * dishes
    included_count = sum(1 for value in included if value)
    totals = result.totals if result.totals_basis != "none" else {}

    cals = int(round(totals.get("calories", 0)))
    p = totals.get("protein", 0)
    c = totals.get("carbohydrates", 0)
    f = totals.get("fats", 0)

    complete = result.totals_basis != "none"
    badge = "Complete" if complete else "Incomplete"
    weight_badge = f'<span class="fm-badge fm-badge-cyan">{_fmt_value(result.total_grams, 0, " g")} Total</span>'

    donut = _macro_donut_svg(cals, p, c, f)
    bar = _macro_bar_html(p, c, f)

    cubes_html = f"""
    <div class="fm-macro-4grid">
      <div class="fm-cube fm-cube-cals"><div class="label">Total Cals</div><div class="val">{cals}</div><div class="sub">kcal</div></div>
      <div class="fm-cube fm-cube-protein"><div class="label">Protein</div><div class="val">{p:.1f}g</div><div class="sub">grams</div></div>
      <div class="fm-cube fm-cube-carbs"><div class="label">Carbs</div><div class="val">{c:.1f}g</div><div class="sub">grams</div></div>
      <div class="fm-cube fm-cube-fats"><div class="label">Fats</div><div class="val">{f:.1f}g</div><div class="sub">grams</div></div>
    </div>
    """

    note_html = ""
    if not complete:
        note = result.totals_reason or "Add weights to compute plate totals."
        note_html = f'<div class="fm-note" style="margin-top:8px;">{html.escape(note)}</div>'

    return (
        '<div class="fm-bento-hero">'
        f'<div class="fm-summary-head"><div class="fm-bento-title">🍱 Plate Summary</div><div style="display:flex;gap:6px;">{weight_badge}<span class="fm-badge">{badge}</span></div></div>'
        '<div style="display: flex; align-items: center; gap: 14px; margin: 4px 0;">'
        f'{donut}'
        f'<div style="flex: 1;">{bar}</div>'
        '</div>'
        f"{cubes_html}"
        f"{note_html}"
        '</div>'
    )


def _single_prediction_html(result) -> str:
    if result is None:
        return (
            '<div class="fm-bento-hero">'
            '<div class="fm-summary-head"><div class="fm-bento-title">🍽️ Dish Prediction</div><span class="fm-badge fm-badge-muted">Standby</span></div>'
            '<div class="fm-bento-standby-row">'
            '<span class="fm-pulse-dot"></span>'
            '<span class="fm-subtle">Ready · Select a sample meal on the left or upload a food photo</span>'
            '</div>'
            '</div>'
        )

    prediction = result.prediction
    if prediction is None or not prediction.available:
        reason = prediction.reason if prediction else "not run"
        return (
            '<div class="fm-bento-hero">'
            '<div class="fm-summary-head"><div class="fm-bento-title">🍽️ Dish Prediction</div>'
            '<span class="fm-badge">Unavailable</span></div>'
            f'<div class="fm-note" style="margin-top:8px;">{html.escape(str(reason))}</div>'
            '</div>'
        )

    confidence = f"{prediction.confidence:.1%}" if prediction.confidence is not None else "—"
    chips = "".join(
        f'<span class="fm-topk-chip"><span class="name">{html.escape(str(name).replace("_", " ").title())}</span><span class="prob">{prob:.0%}</span></span>'
        for name, prob in (prediction.top_k or [])
    )
    chips_html = f'<div class="fm-topk-chips">{chips}</div>' if chips else ""
    return (
        '<div class="fm-bento-hero">'
        f'<div class="fm-summary-head"><div class="fm-bento-title">🍽️ Dish Prediction</div><span class="fm-badge fm-badge-green">{confidence} Match</span></div>'
        f'<div class="fm-pred-name" style="margin:4px 0 6px;">{html.escape(prediction.label.replace("_", " ").title())}</div>'
        f"{chips_html}"
        '</div>'
    )


def _single_weight_html(result) -> str:
    if result is None:
        return (
            '<div class="fm-bento-hero">'
            '<div class="fm-weight-strip">'
            '<div class="fm-weight-left">'
            '<div class="fm-bento-title">⚖️ Portion Mass</div>'
            '<div class="fm-subtle">Portion weight appears after scan</div>'
            '</div>'
            '<span class="fm-badge fm-badge-muted">Standby</span>'
            '</div>'
            '</div>'
        )

    weight = result.weight
    if weight is None or not weight.available:
        reason = "; ".join(weight.warnings) if weight and weight.warnings else "no weight source"
        return (
            '<div class="fm-bento-hero">'
            '<div class="fm-weight-strip">'
            '<div class="fm-weight-left">'
            '<div class="fm-bento-title">⚖️ Portion Mass & Serving Basis</div>'
            f'<div class="fm-note" style="margin-top:2px;">{html.escape(reason)}</div>'
            '</div>'
            '<div class="fm-weight-right">'
            '<span class="fm-badge fm-badge-amber">Unavailable</span>'
            '<span class="fm-weight-basis-pill">Reference: 100g</span>'
            '</div>'
            '</div>'
            '</div>'
        )

    status = weight.status.capitalize()
    weight_badge_cls = "fm-badge fm-badge-cyan" if weight.status == "estimated" else "fm-badge"
    return (
        '<div class="fm-bento-hero">'
        '<div class="fm-weight-strip">'
        '<div class="fm-weight-left">'
        '<div class="fm-bento-title">⚖️ Portion Mass</div>'
        f'<div class="fm-note" style="margin-top:2px;">{html.escape(weight.method)}</div>'
        '</div>'
        '<div class="fm-weight-right">'
        '<div class="fm-weight-hero" style="margin:0;display:inline-flex;align-items:baseline;gap:3px;">'
        f'<span class="fm-weight-val" style="font-size:20px;font-weight:900;color:var(--fm-ink);">{weight.grams:.1f}</span>'
        '<span class="fm-weight-unit" style="font-size:12px;font-weight:700;color:var(--fm-subtext);">g</span>'
        '</div>'
        f'<span class="{weight_badge_cls}">{html.escape(status)}</span>'
        '</div>'
        '</div>'
        '</div>'
    )


def _single_nutrition_html(result) -> str:
    if result is None:
        return (
            '<div class="fm-bento-hero">'
            '<div class="fm-summary-head"><div class="fm-bento-title">⚡ Macronutrients</div><span class="fm-badge fm-badge-muted">—</span></div>'
            '<div class="fm-bento-standby-row"><span class="fm-subtle">Macro breakdown appears after scan</span></div>'
            '</div>'
        )

    nutrition = result.nutrition
    if nutrition is None or not nutrition.available:
        reason = nutrition.reason if nutrition else "not run"
        return (
            '<div class="fm-bento-hero">'
            '<div class="fm-summary-head"><div class="fm-bento-title">⚡ Macronutrients</div>'
            '<span class="fm-badge">Unavailable</span></div>'
            f'<div class="fm-note" style="margin-top:8px;">{html.escape(str(reason))}</div>'
            '</div>'
        )

    nutrients = nutrition.nutrients or {}
    cals = int(round(nutrients.get("calories", 0)))
    p = nutrients.get("protein", 0)
    c = nutrients.get("carbohydrates", 0)
    f = nutrients.get("fats", 0)

    cubes_html = f"""
    <div class="fm-macro-4grid">
      <div class="fm-cube fm-cube-cals">
        <div class="label">Calories</div>
        <div class="val">{cals}</div>
        <div class="sub">kcal</div>
      </div>
      <div class="fm-cube fm-cube-protein">
        <div class="label">Protein</div>
        <div class="val">{p:.1f}g</div>
        <div class="sub">grams</div>
      </div>
      <div class="fm-cube fm-cube-carbs">
        <div class="label">Carbs</div>
        <div class="val">{c:.1f}g</div>
        <div class="sub">grams</div>
      </div>
      <div class="fm-cube fm-cube-fats">
        <div class="label">Fats</div>
        <div class="val">{f:.1f}g</div>
        <div class="sub">grams</div>
      </div>
    </div>
    """

    bar_html = _macro_bar_html(p, c, f)

    micro_chips = []
    if "fiber" in nutrients:
        micro_chips.append(f'<span class="fm-micro-chip">🌾 Fiber <strong>{nutrients["fiber"]:.1f}g</strong></span>')
    if "sugars" in nutrients:
        micro_chips.append(f'<span class="fm-micro-chip">🍬 Sugar <strong>{nutrients["sugars"]:.1f}g</strong></span>')
    if "sodium" in nutrients:
        micro_chips.append(f'<span class="fm-micro-chip">🧂 Sodium <strong>{int(nutrients["sodium"])}mg</strong></span>')
    micro_html = f'<div class="fm-micro-chips">{"".join(micro_chips)}</div>' if micro_chips else ""

    basis = f"{nutrition.basis} · {_fmt_value(nutrition.grams, 0, ' g')}"
    return (
        '<div class="fm-bento-hero">'
        f'<div class="fm-summary-head"><div class="fm-bento-title">⚡ Macronutrients</div><span class="fm-badge">{html.escape(basis)}</span></div>'
        f"{cubes_html}"
        f"{bar_html}"
        f"{micro_html}"
        '</div>'
    )


_DISH_MACRO_META = [
    meta
    for meta in _NUTRITION_META
    if meta[0] in {"calories", "protein", "carbohydrates", "fats"}
]


def _per_dish_nutrition_html(result, reason=None) -> str:
    if result is None:
        note = html.escape(reason or "Scan a plate to see per-dish calories and macros.")
        return (
            '<div class="fm-bento-hero">'
            '<div class="fm-summary-head"><div class="fm-bento-title">🍽️ Nutrition by Dish</div><span class="fm-badge fm-badge-muted">Standby</span></div>'
            '<div class="fm-bento-standby-row">'
            '<span class="fm-pulse-dot"></span>'
            f'<span class="fm-subtle">{note}</span>'
            '</div>'
            '</div>'
        )

    if not result.dishes:
        return (
            '<div class="fm-bento-hero">'
            '<div class="fm-summary-head"><div class="fm-bento-title">🍽️ Nutrition by Dish</div></div>'
            '<div class="fm-note" style="margin-top:8px;">No dishes were detected.</div>'
            '</div>'
        )

    BOX_PALETTE = ['#10b981', '#06b6d4', '#f59e0b', '#8b5cf6', '#f43f5e', '#3b82f6', '#eab308']
    included = list(result.included) if result.included else [True] * len(result.dishes)
    cards = []
    for dish, is_included in zip(result.dishes, included):
        prediction = dish.prediction
        confidence_text = ""
        if prediction is not None and prediction.available and prediction.label:
            name = prediction.label.replace("_", " ").title()
            if prediction.confidence is not None:
                confidence_text = f"{prediction.confidence:.0%}"
        elif dish.detection is not None and dish.detection.label:
            name = dish.detection.label.replace("_", " ").title()
            if dish.detection.confidence is not None:
                confidence_text = f"{dish.detection.confidence:.0%}"
        else:
            name = f"Food {dish.index}"

        weight = dish.weight
        if weight is not None and weight.available:
            weight_text = f"{_fmt_value(weight.grams, 0, ' g')} · {weight.status}"
            weight_badge_cls = "fm-badge fm-badge-cyan" if weight.status == "estimated" else "fm-badge"
        else:
            weight_text = "Weight unavailable"
            weight_badge_cls = "fm-badge fm-badge-muted"

        nutrients = dish.nutrition.nutrients if (dish.nutrition and dish.nutrition.available) else {}
        cal_str = _fmt_value(nutrients.get("calories"), 0, " kcal") if nutrients else "Calories unavailable"
        p_str = _fmt_value(nutrients.get("protein"), 1, "g") if nutrients else "—"
        c_str = _fmt_value(nutrients.get("carbohydrates"), 1, "g") if nutrients else "—"
        f_str = _fmt_value(nutrients.get("fats"), 1, "g") if nutrients else "—"

        color = BOX_PALETTE[(dish.index - 1) % len(BOX_PALETTE)]

        cards.append(
            f'<article class="fm-dish-card" style="border-left: 3px solid {color};">'
            '<div class="fm-dish-head" style="display:flex;align-items:center;justify-content:space-between;gap:8px;">'
            f'<div style="display:flex;align-items:center;gap:8px;"><div class="fm-dish-num-badge" style="background:{color}25; color:{color}; border-color:{color};">#{dish.index}</div>'
            f'<div class="fm-dish-name" style="font-size:15px;font-weight:800;color:var(--fm-ink);">{html.escape(name)}</div></div>'
            f'<span class="fm-badge fm-badge-purple">{confidence_text or "Match"}</span>'
            '</div>'
            '<div class="fm-dish-mini-row" style="margin:4px 0;">'
            f'<span class="{weight_badge_cls}">{html.escape(weight_text)}</span>'
            '</div>'
            '<div class="fm-dish-macro-chips">'
            f'<span class="chip-cal">🔥 Calories: {cal_str}</span>'
            f'<span class="chip-p">P {p_str}</span>'
            f'<span class="chip-c">C {c_str}</span>'
            f'<span class="chip-f">F {f_str}</span>'
            '</div>'
            '</article>'
        )

    dish_word = "dish" if len(result.dishes) == 1 else "dishes"
    return (
        '<div class="fm-summary-head">'
        f'<div class="fm-bento-title">🍽️ Nutrition by Dish</div><span class="fm-badge">{len(result.dishes)} {dish_word}</span>'
        '</div>'
        f'<div class="fm-dish-grid">{"".join(cards)}</div>'
    )


def _image_input(gr, label, height=None):
    kwargs = {"type": "filepath", "sources": ["upload", "webcam"], "label": label}
    if height is not None:
        kwargs["height"] = height
    try:
        return gr.Image(type="filepath", sources=["upload", "webcam"], label=label)
    except TypeError:
        return gr.Image(type="filepath", label=label)


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "y", "x", "include")
    return bool(value)


def _as_grams(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        grams = float(text)
    except (TypeError, ValueError):
        return None
    return grams if grams > 0 else None


def _table_rows(table):
    if table is None:
        return []
    if hasattr(table, "values"):  # pandas DataFrame
        return [list(row) for row in table.values.tolist()]
    if hasattr(table, "tolist"):
        return table.tolist()
    return list(table)


def multi_detect(multi_pipeline, image, top_k, max_dishes, crop_padding):
    if image is None:
        return None, None, [], "No image provided.", {}, LIMITATIONS
    result = multi_pipeline.analyze(
        image,
        top_k=int(top_k),
        max_dishes=int(max_dishes),
        crop_padding=float(crop_padding),
    )
    if not result.available:
        return None, None, [], f"Analysis unavailable: {result.reason}", {}, LIMITATIONS

    detections = [d.detection for d in result.dishes if d.detection is not None]
    labels = []
    for d in result.dishes:
        if d.detection is not None:
            pred_name = d.predicted_label or d.detection.label or f"Dish {d.index}"
            conf_str = f" ({d.prediction.confidence:.0%})" if (d.prediction and d.prediction.confidence) else ""
            labels.append(f"#{d.index} {pred_name.title()}{conf_str}")

    try:
        annotated = draw_detections(image, detections, labels=labels) if detections else image
        if hasattr(annotated, "convert"):
            import numpy as np
            annotated = np.asarray(annotated.convert("RGB"))
    except Exception:
        annotated = image

    rows = []
    for dish in result.dishes:
        det = dish.detection
        pred = dish.prediction
        rows.append(
            [
                dish.index,
                det.label if det else "whole image",
                pred.label if (pred is not None and pred.available) else "",
                (
                    f"{pred.confidence:.1%}"
                    if (pred is not None and pred.available and pred.confidence is not None)
                    else ""
                ),
                True,
                None,
            ]
        )
    return (
        annotated,
        result,
        rows,
        format_multi_analysis(result),
        multi_analysis_to_dict(result),
        LIMITATIONS,
    )


def multi_compute(multi_pipeline, state, table):
    """Read the dish table's Include/Grams columns and recompute nutrition."""
    if state is None:
        return (
            "No detection result yet. Click '1) Detect & classify' first.",
            {},
            LIMITATIONS,
        )
    rows = _table_rows(table)
    grams_by_index = {}
    include_by_index = {}
    for row in rows:
        if not row:
            continue
        try:
            idx = int(float(row[0]))
        except (TypeError, ValueError, IndexError):
            continue
        include_by_index[idx] = _as_bool(row[4]) if len(row) > 4 else True
        grams_by_index[idx] = _as_grams(row[5]) if len(row) > 5 else None

    per_dish_grams = [grams_by_index.get(d.index) for d in state.dishes]
    include = [include_by_index.get(d.index, True) for d in state.dishes]
    updated = multi_pipeline.apply_weights(state, per_dish_grams, include)
    return format_multi_analysis(updated), multi_analysis_to_dict(updated), LIMITATIONS


def build_interface(pipeline: FoodAnalysisPipeline, multi_pipeline=None):
    import gradio as gr

    def run(image, measured, top_k):
        if image is None:
            return "No image provided.", "", "", "", {}, LIMITATIONS
        result = pipeline.analyze(
            image, measured_grams=_clean_measured(measured), top_k=int(top_k)
        )
        notes = "\n".join(f"- {w}" for w in result.warnings) or "No warnings."
        return (
            _classification_text(result),
            result.detection.summary() if result.detection else "not run",
            result.weight.summary(),
            _macros_text(result),
            analysis_to_dict(result),
            notes + "\n\n" + LIMITATIONS,
        )

    def run_multi_detect(image, top_k, max_dishes, crop_padding):
        return multi_detect(multi_pipeline, image, top_k, max_dishes, crop_padding)

    theme = _build_theme(gr)
    
    with gr.Blocks(title="FoodMacros Vision AI", theme=theme, css=CSS) as demo:
        # Header Row
        with gr.Row(elem_classes="fm-header-row"):
            gr.HTML(BRAND_HTML)

        # Main Navigation Tabs (Clean, prominent tabs)
        with gr.Tabs(elem_classes="fm-main-tabs") as main_tabs:
            
            # TAB 1: Single Dish Scanner
            with gr.Tab("🍽️ Single Dish Scanner", id="single_tab"):
                _single_dish_tab(pipeline, run, gr)

            # TAB 2: Multi-Dish Plate Scanner
            with gr.Tab("🍱 Multi-Dish (Plate Scanner)", id="multi_tab"):
                if multi_pipeline is not None:
                    _multi_dish_tab(multi_pipeline, run_multi_detect, gr)
                else:
                    with gr.Column(elem_classes="fm-card"):
                        gr.Markdown("Multi-dish pipeline is not available.")

            # TAB 3: About
            with gr.Tab("ℹ️ About & Transparency", id="about_tab"):
                _about_tab(gr)

    demo.foodmacros_theme = theme
    demo.foodmacros_css = CSS
    return demo


def _single_dish_tab(pipeline: FoodAnalysisPipeline, run, gr):
    # Row 1: Balanced 2-Column Scanner Arena (Image & Controls on Left, AI Bento on Right)
    with gr.Row(equal_height=False, elem_classes="fm-layout-row"):
        # Left Column: Input Viewport & Controls
        with gr.Column(scale=5, min_width=320, elem_classes="fm-card"):
            gr.HTML(
                _section_html(
                    "Single Dish Scanner",
                    "Scan or upload one food item to classify it and estimate its portion weight and macros.",
                )
            )
            image_input = _image_input(gr, "Food Image", height=280)
            
            with gr.Row(elem_classes="fm-compact-controls"):
                measured_input = gr.Number(
                    value=None,
                    label="Scale Weight (g, opt)",
                    precision=1,
                    scale=1,
                )
                top_k = gr.Slider(
                    1,
                    5,
                    value=int(pipeline.config.ui.max_top_k),
                    step=1,
                    label="Top-k Predictions",
                    scale=1,
                )

            analyze_button = gr.Button(
                "⚡ Analyze Dish", variant="primary", elem_classes="fm-primary"
            )

        # Right Column: High-Tech HUD Intelligence Cards (Full-width clean stack!)
        with gr.Column(scale=6, min_width=340):
            with gr.Column(elem_classes="fm-card fm-summary"):
                prediction_out = gr.HTML(_single_prediction_html(None))

            with gr.Column(elem_classes="fm-card fm-weight-card"):
                weight_out_html = gr.HTML(_single_weight_html(None))

            with gr.Column(elem_classes="fm-card fm-nutrition-card"):
                nutrition_out = gr.HTML(_single_nutrition_html(None))

    # Row 2: 1-Click Samples spanning FULL WIDTH (no black void on right!)
    sample_dir = PROJECT_ROOT / "assets" / "samples"
    sample_files = [
        sample_dir / "sample_salad.png",
        sample_dir / "sample_meal.png",
        sample_dir / "sample_plate.png",
    ]
    valid_samples = [[str(p), None, 5] for p in sample_files if p.exists()]
    if valid_samples:
        with gr.Row(elem_classes="fm-samples-fullrow"):
            with gr.Column(elem_classes="fm-card"):
                gr.Examples(
                    examples=valid_samples,
                    inputs=[image_input, measured_input, top_k],
                    label="✨ 1-Click Sample Meals · Click any meal to load and test immediately",
                )

    # Row 3: Technical Details full width
    with gr.Accordion("🛠️ Technical Diagnostics & Data Stream", open=False):
        cls_out = gr.Textbox(label="Classification Log", lines=4)
        det_out = gr.Textbox(label="Detection Bounds", lines=2)
        weight_out = gr.Textbox(label="Portion Weight Status", lines=2)
        macros_out = gr.Textbox(label="Macronutrient Payload", lines=8)
        json_out = gr.JSON(label="Structured Result")
        notes_out = gr.Markdown(value=LIMITATIONS, elem_classes="fm-limitations")

    analyze_button.click(
        run,
        inputs=[image_input, measured_input, top_k],
        outputs=[cls_out, det_out, weight_out, macros_out, json_out, notes_out],
    )


def _multi_dish_tab(multi_pipeline: MultiDishPipeline, run_multi_detect, run_multi_compute, gr):
    det_cfg = multi_pipeline.config.detection

    with gr.Row(equal_height=False, elem_classes="fm-layout-row"):
        # Left Column: Vision Processing & Command Cockpit
        with gr.Column(scale=5, min_width=320, elem_classes="fm-card"):
            gr.HTML(
                _section_html(
                    "Plate Vision Cockpit",
                    "Dual-viewport neural scanner localizing individual dishes via YOLOv8.",
                )
            )
            # Symmetrical Side-by-Side Dual Viewports inside Left Cockpit
            with gr.Row(equal_height=True, elem_classes="fm-image-row"):
                with gr.Column(scale=1, min_width=0):
                    gr.HTML(_section_html("Source Plate"))
                    image_input = _image_input(gr, "Upload plate photo", height=250)
                with gr.Column(scale=1, min_width=0):
                    gr.HTML(_section_html("YOLOv8 Detections"))
                    annotated_out = gr.Image(
                        label="Detected dishes",
                        interactive=False,
                        height=250,
                        elem_classes="fm-image",
                    )

            # Compact Control Strip
            with gr.Row(elem_classes="fm-compact-controls"):
                top_k = gr.Slider(
                    1,
                    5,
                    value=int(multi_pipeline.config.ui.max_top_k),
                    step=1,
                    label="Top-k per dish",
                    scale=1,
                )
                max_dishes = gr.Slider(
                    1,
                    10,
                    value=int(det_cfg.max_dishes),
                    step=1,
                    label="Max dishes",
                    scale=1,
                )
                crop_padding = gr.Slider(
                    0.0,
                    0.3,
                    value=float(det_cfg.crop_padding),
                    step=0.01,
                    label="Box Padding",
                    scale=1,
                )

            detect_button = gr.Button(
                "⚡ Scan & Localize Plate (YOLOv8)", variant="primary", elem_classes="fm-primary fm-scan-btn"
            )

            # 1-Click Samples for Multi-Dish Plate (directly accessible under scan button!)
            sample_dir = PROJECT_ROOT / "assets" / "samples"
            multi_sample_files = [
                sample_dir / "sample_plate.png",
                sample_dir / "sample_meal.png",
                sample_dir / "sample_salad.png",
            ]
            valid_multi_samples = [[str(p), 5, 8, 0.08] for p in multi_sample_files if p.exists()]
            if valid_multi_samples:
                gr.Examples(
                    examples=valid_multi_samples,
                    inputs=[image_input, top_k, max_dishes, crop_padding],
                    label="✨ 1-Click Sample Plates · Click any plate to test immediately",
                )

        # Right Column: High-Tech Nutrition Intelligence Deck (Summary right at eye-level!)
        with gr.Column(scale=6, min_width=340):
            # Plate Nutrition Summary (Prominently placed at eye-level, NOT downwards!)
            with gr.Column(elem_classes="fm-card fm-summary"):
                summary_out = gr.HTML(_plate_summary_html(None))

            # Per-Dish Breakdown Bento Grid
            with gr.Column(elem_classes="fm-card"):
                per_dish_out = gr.HTML(_per_dish_nutrition_html(None))

    # Full-width Collapsible Technical Diagnostics below both columns
    with gr.Accordion("🛠️ Technical Diagnostics & Data Stream", open=False):
        report_out = gr.Textbox(label="Raw analysis report", lines=16)
        json_out = gr.JSON(label="Structured result")
        notes_out = gr.Markdown(value=LIMITATIONS, elem_classes="fm-limitations")

    detect_button.click(
        run_multi_detect,
        inputs=[image_input, top_k, max_dishes, crop_padding],
        outputs=[
            annotated_out,
            summary_out,
            per_dish_out,
            report_out,
            json_out,
            notes_out,
        ],
    )


def _about_tab(gr):
    with gr.Column(elem_classes="fm-card"):
        gr.HTML(
            _section_html(
                "About FoodMacros Vision AI",
                "FoodMacros combines neural food classification, YOLO localization, portion-weight "
                "estimation, and nutrition lookup in one unified dashboard.",
            )
        )
        gr.Markdown(
            """
### How it works

1. **Localize**: The YOLOv8 detector scans the plate to isolate individual food regions.
2. **Crop & Classify**: High-resolution crops are fed into EfficientNetB0 to identify the dish across 101 food categories.
3. **Estimate Weight**: A trained MobileNetV2 regressor estimates portion mass (or applies user-measured scale readings).
4. **Calculate Macros**: Macronutrients (Calories, Protein, Carbs, Fats, Fiber, Sugars, Sodium) are looked up and scaled proportionally.

### Reliability & Transparency

Portion mass is strictly categorized into:
* **Measured**: Scaled on a physical scale (Ground truth).
* **Estimated**: Vision-based regression model prediction.
* **Unavailable**: When no weight model or measurement is supplied, the system honestly falls back to a standardized 100g reference serving rather than hallucinating false weights.
""",
            elem_classes="fm-about-copy",
        )
        gr.Markdown(LIMITATIONS, elem_classes="fm-limitations")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="FoodMacros Gradio app")
    parser.add_argument("--config", default=None, help="Path to a config JSON")
    parser.add_argument("--model-dir", default=None, help="Directory with classifier.keras")
    parser.add_argument("--regressor-dir", default=None, help="Directory with mass_regressor.keras")
    parser.add_argument("--share", action="store_true", help="Create a public share link")
    parser.add_argument("--port", type=int, default=None, help="Server port")
    args = parser.parse_args(argv)

    config = load_config(args.config) if args.config else load_config()
    if args.share:
        config.ui.share = True
    if args.port is not None:
        config.ui.server_port = args.port

    if not has_package("gradio"):
        print(
            "gradio is not installed. Install it with 'pip install gradio' to run the app.\n"
            "The rest of the project (classification, detection, weight, nutrition) does not "
            "require gradio."
        )
        return 1

    pipeline, resolved = create_pipeline(config, args.model_dir, args.regressor_dir)
    print(resolved.summary())
    for note in pipeline.notes:
        print(f"note: {note}")

    multi_pipeline = create_multi_pipeline(config, args.model_dir)
    for note in multi_pipeline.notes:
        print(f"note: {note}")

    demo = build_interface(pipeline, multi_pipeline=multi_pipeline)
    launch_kwargs = {
        "share": config.ui.share,
        "server_port": config.ui.server_port,
        "theme": getattr(demo, "foodmacros_theme", _build_theme(gr)),
        "css": getattr(demo, "foodmacros_css", CSS),
        "allowed_paths": [str(PROJECT_ROOT)],
    }
    try:
        demo.launch(**launch_kwargs)
    except TypeError:
        launch_kwargs.pop("theme", None)
        launch_kwargs.pop("css", None)
        try:
            demo.launch(**launch_kwargs)
        except Exception as exc:  # tunnel/network failures must not crash the shell
            print(f"Failed to launch Gradio: {exc}")
            return 1
    except Exception as exc:  # tunnel/network failures must not crash the shell
        print(f"Failed to launch Gradio: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
