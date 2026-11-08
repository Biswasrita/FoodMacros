"""Interactive Gradio application for the FoodMacros pipeline.

Gradio is imported lazily so the rest of the project works without it. Launch
locally by default; ``--share`` (or ``ui.share`` in config) enables a public
tunnel and failures are reported gracefully instead of crashing.
"""

from __future__ import annotations

import argparse
import html
import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from foodmacros.config import AppConfig, load_config  # noqa: E402
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
    NUTRIENT_LABELS,
    analysis_to_dict,
    format_nutrients,
)


BRAND_HTML = """
<div class="fm-brand">
  <div class="fm-logo" aria-hidden="true">
    <svg viewBox="0 0 24 24" width="25" height="25" fill="none" stroke="currentColor"
         stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">
      <path d="M20 4C10.5 4 4 9.2 4 17c0 2 1 3 3 3 8.2 0 13-6.5 13-16Z"/>
      <path d="M4 20c3.2-5.2 7.2-8.4 12.2-10.4"/>
    </svg>
  </div>
  <div>
    <div class="fm-title">FoodMacros</div>
    <div class="fm-subtitle">AI-Powered Food Analysis &amp; Nutrition Estimation</div>
  </div>
</div>
"""

CSS = """
:root {
  --fm-green-950: #0b2e1c;
  --fm-green-900: #14532d;
  --fm-green-700: #166534;
  --fm-green-500: #22c55e;
  --fm-green-100: #dcfce7;
  --fm-green-50: #f0fdf4;
  --fm-ink: #0f172a;
  --fm-body: #1e293b;
  --fm-muted: #475569;
  --fm-border: #e2e8f0;
  --fm-bg: #f5f7f4;
}

body, .gradio-container {
  background: var(--fm-bg) !important;
}

.gradio-container {
  max-width: 1500px !important;
  margin: 0 auto !important;
  padding: 20px !important;
  color: var(--fm-body) !important;
  font-family: "Segoe UI", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif !important;
  font-size: 15px;
  line-height: 1.6;
}

.gradio-container button,
.gradio-container input,
.gradio-container textarea,
.gradio-container select,
.gradio-container table {
  font-family: inherit !important;
}

.fm-nav {
  align-items: center !important;
  background: #ffffff !important;
  border: 1px solid var(--fm-border) !important;
  border-radius: 20px !important;
  padding: 14px 22px !important;
  box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06) !important;
  margin-bottom: 18px !important;
  gap: 16px !important;
}

.fm-brand {
  display: flex;
  align-items: center;
  gap: 14px;
}

.fm-logo {
  width: 46px;
  height: 46px;
  border-radius: 14px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #ffffff;
  background: linear-gradient(145deg, var(--fm-green-900), var(--fm-green-500));
  box-shadow: 0 8px 18px rgba(20, 83, 45, 0.25);
  flex: 0 0 auto;
}

.fm-title {
  font-size: 22px;
  font-weight: 800;
  letter-spacing: -0.02em;
  color: var(--fm-ink);
  line-height: 1.1;
}

.fm-subtitle {
  font-size: 12px;
  color: var(--fm-muted);
  font-weight: 600;
  margin-top: 2px;
}

.fm-nav-radio {
  justify-content: flex-end;
}

.fm-nav-radio .wrap {
  justify-content: flex-end;
  gap: 6px !important;
}

.fm-nav-radio label {
  border-radius: 999px !important;
  padding: 9px 18px !important;
  margin: 0 !important;
  background: #f1f5f2 !important;
  border: 1px solid transparent !important;
  color: var(--fm-green-900) !important;
  font-weight: 700 !important;
  cursor: pointer;
  transition: all 0.18s ease;
}

.fm-nav-radio label span {
  color: inherit !important;
}

.fm-nav-radio label:has(input:checked) {
  background: var(--fm-green-900) !important;
  color: #ffffff !important;
  border-color: var(--fm-green-900) !important;
  box-shadow: 0 6px 16px rgba(20, 83, 45, 0.25);
}

.fm-nav-radio label:has(input:checked) * {
  color: #ffffff !important;
}

.fm-card {
  background: #ffffff !important;
  border: 1px solid var(--fm-border) !important;
  border-radius: 20px !important;
  padding: 20px !important;
  box-shadow: 0 12px 34px rgba(15, 23, 42, 0.05) !important;
  color: var(--fm-body) !important;
}

.fm-card .prose,
.fm-card .prose p,
.fm-card .prose li,
.fm-about-copy .prose,
.fm-about-copy .prose p,
.fm-about-copy .prose li {
  color: var(--fm-body) !important;
  font-size: 15px !important;
  line-height: 1.65 !important;
}

.fm-card .prose strong,
.fm-about-copy .prose strong {
  color: var(--fm-ink) !important;
  font-weight: 800 !important;
}

.fm-card .prose h1,
.fm-card .prose h2,
.fm-card .prose h3,
.fm-about-copy .prose h1,
.fm-about-copy .prose h2,
.fm-about-copy .prose h3 {
  color: var(--fm-green-900) !important;
  font-weight: 800 !important;
  letter-spacing: -0.01em;
}

.fm-about-copy .prose h3 {
  font-size: 18px !important;
  margin: 18px 0 8px !important;
}

.fm-about-copy .prose ol,
.fm-about-copy .prose ul {
  padding-left: 22px !important;
}

.fm-about-copy .prose li {
  margin: 6px 0 !important;
}

.fm-card label > span,
.fm-card .block-info {
  color: #334155 !important;
  background: transparent !important;
  font-weight: 600 !important;
}

.fm-card input,
.fm-card textarea {
  color: var(--fm-ink) !important;
  font-weight: 500 !important;
}

.fm-card .label-wrap,
.fm-card .label-wrap span,
.gradio-container .label-wrap,
.gradio-container .label-wrap span {
  color: var(--fm-ink) !important;
  font-weight: 700 !important;
}

.gradio-container .block label > span,
.gradio-container .block .block-info {
  color: #334155 !important;
  background: transparent !important;
  font-weight: 600 !important;
}

.gradio-container .block input,
.gradio-container .block textarea {
  color: var(--fm-ink) !important;
  font-weight: 500 !important;
}

.fm-limitations .prose {
  color: var(--fm-muted) !important;
  font-size: 13px !important;
  line-height: 1.6 !important;
}

.fm-summary {
  background: linear-gradient(160deg, #ffffff 0%, var(--fm-green-50) 100%) !important;
}

.fm-section-title h2,
.fm-section-title h3 {
  margin: 0;
  color: var(--fm-ink);
  letter-spacing: -0.02em;
}

.fm-section-title h2 {
  font-size: 20px;
}

.fm-section-title h3 {
  font-size: 17px;
}

.fm-section-title p {
  margin: 6px 0 0;
}

.fm-summary-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}

.fm-summary-head h3 {
  margin: 0;
  font-size: 17px;
  color: var(--fm-ink);
  letter-spacing: -0.01em;
}

.fm-badge {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  background: var(--fm-green-100);
  color: var(--fm-green-900);
  font-size: 12px;
  font-weight: 800;
  white-space: nowrap;
}

.fm-badge-muted {
  background: #eef2f0;
  color: #334155;
}

.fm-dish-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
  gap: 12px;
  margin-top: 14px;
}

.fm-dish-card {
  background: #ffffff;
  border: 1px solid var(--fm-border);
  border-radius: 16px;
  padding: 14px;
}

.fm-dish-card.fm-dish-excluded {
  background: #f8faf9;
  border-style: dashed;
}

.fm-dish-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 10px;
}

.fm-dish-name {
  color: var(--fm-ink);
  font-size: 16px;
  font-weight: 800;
  line-height: 1.3;
  text-transform: capitalize;
}

.fm-dish-badges {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 6px;
}

.fm-dish-macros {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px;
  margin-top: 12px;
}

.fm-dish-macro {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 8px 10px;
  background: #f8faf9;
  border: 1px solid #e2e8e4;
  border-radius: 10px;
}

.fm-dish-macro span {
  display: flex;
  align-items: center;
  color: var(--fm-muted);
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.fm-dish-macro strong {
  color: var(--fm-ink);
  font-size: 15px;
  font-weight: 800;
}

.fm-dish-note {
  margin-top: 10px;
}

.fm-pred-name {
  font-size: 28px;
  line-height: 1.15;
  font-weight: 800;
  color: var(--fm-green-900);
  letter-spacing: -0.03em;
  text-transform: capitalize;
}

.fm-note {
  color: var(--fm-muted);
  font-size: 13px;
  line-height: 1.55;
}

.fm-stat-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
  margin-top: 14px;
}

.fm-stat {
  background: #f8faf9;
  border: 1px solid #e2e8e4;
  border-radius: 14px;
  padding: 12px 14px;
}

.fm-stat .fm-stat-label {
  display: flex;
  align-items: center;
  font-size: 11px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--fm-muted);
  font-weight: 800;
}

.fm-stat .fm-stat-value {
  font-size: 20px;
  font-weight: 800;
  color: var(--fm-ink);
  margin-top: 4px;
  letter-spacing: -0.02em;
}

.fm-stat .fm-stat-value small {
  font-size: 11px;
  font-weight: 700;
  color: var(--fm-muted);
}

.fm-kv {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  padding: 7px 0;
  border-bottom: 1px solid #edf1ee;
  font-size: 13px;
}

.fm-kv:last-child {
  border-bottom: 0;
}

.fm-kv span:first-child {
  color: var(--fm-muted);
  font-weight: 600;
}

.fm-kv span:last-child {
  color: var(--fm-ink);
  font-weight: 800;
  text-align: right;
}

.fm-image,
.fm-image .image-container,
.fm-image img {
  border-radius: 14px !important;
}

.fm-primary {
  background: var(--fm-green-900) !important;
  border-color: var(--fm-green-900) !important;
  color: #ffffff !important;
  font-weight: 700 !important;
}

.fm-primary:hover {
  background: var(--fm-green-700) !important;
  border-color: var(--fm-green-700) !important;
}

@media (max-width: 900px) {
  .gradio-container {
    padding: 12px !important;
  }

  .fm-nav {
    flex-direction: column !important;
    align-items: flex-start !important;
  }

  .fm-nav-radio,
  .fm-nav-radio .wrap {
    justify-content: flex-start !important;
    flex-wrap: wrap;
  }

  .fm-stat-grid,
  .fm-dish-macros {
    grid-template-columns: 1fr;
  }

  .fm-layout-row,
  .fm-image-row {
    flex-direction: column !important;
  }

  .fm-layout-row > *,
  .fm-image-row > * {
    width: 100% !important;
  }

  .fm-pred-name {
    font-size: 23px;
  }
}
"""


def create_pipeline(config: AppConfig, model_dir=None, regressor_dir=None):
    pipeline, resolved = FoodAnalysisPipeline.from_discovery(
        config, model_dir=model_dir, mass_regressor_dir=regressor_dir
    )
    return pipeline, resolved


def create_multi_pipeline(config: AppConfig, model_dir=None, regressor_dir=None):
    """Build the multi-dish pipeline from the project's default artifacts."""
    return MultiDishPipeline.from_defaults(
        config, model_dir=model_dir, mass_regressor_dir=regressor_dir
    )


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


def _plate_summary_html(result, reason=None) -> str:
    if result is None:
        note = html.escape(reason or "Analyze a plate to see its weight and nutrition summary.")
        return (
            '<div class="fm-summary-head"><h3>Plate Summary</h3></div>'
            f'<div class="fm-note" style="margin-top:12px;">{note}</div>'
        )

    dishes = len(result.dishes)
    included = list(result.included) if result.included else [True] * dishes
    included_count = sum(1 for value in included if value)
    totals = result.totals if result.totals_basis != "none" else {}

    dishes_value = str(dishes)
    if included_count != dishes:
        dishes_value += f' <small>{included_count} included</small>'

    stats = "".join(
        [
            _stat_html("Dishes Detected", dishes_value, "dishes"),
            _stat_html("Total Food Weight", _fmt_value(result.total_grams, 0, " g"), "weight"),
            _stat_html("Calories", _fmt_value(totals.get("calories"), 0, " kcal"), "flame"),
            _stat_html("Protein", _fmt_value(totals.get("protein"), 1, " g"), "protein"),
            _stat_html("Carbs", _fmt_value(totals.get("carbohydrates"), 1, " g"), "carbs"),
            _stat_html("Fat", _fmt_value(totals.get("fats"), 1, " g"), "fat"),
        ]
    )

    complete = result.totals_basis != "none"
    badge = "Complete" if complete else "Incomplete"
    note_html = ""
    if not complete:
        note = result.totals_reason or "Add weights to compute plate totals."
        note_html = f'<div class="fm-note" style="margin-top:14px;">{html.escape(note)}</div>'

    return (
        '<div class="fm-summary-head">'
        f"<h3>Plate Summary</h3><span class=\"fm-badge\">{badge}</span>"
        "</div>"
        f'<div class="fm-stat-grid">{stats}</div>'
        f"{note_html}"
    )


def _single_prediction_html(result) -> str:
    if result is None:
        return (
            '<div class="fm-summary-head"><h3>Prediction</h3></div>'
            '<div class="fm-note" style="margin-top:12px;">Upload a food image and click '
            "Analyze to see the predicted dish.</div>"
        )

    prediction = result.prediction
    if prediction is None or not prediction.available:
        reason = prediction.reason if prediction else "not run"
        return (
            '<div class="fm-summary-head"><h3>Prediction</h3>'
            '<span class="fm-badge">Unavailable</span></div>'
            f'<div class="fm-note" style="margin-top:12px;">{html.escape(str(reason))}</div>'
        )

    confidence = (
        f"{prediction.confidence:.1%}" if prediction.confidence is not None else "—"
    )
    top = "".join(
        f'<div class="fm-kv"><span>{html.escape(str(name))}</span>'
        f"<span>{probability:.1%}</span></div>"
        for name, probability in prediction.top_k
    )
    top_html = f'<div style="margin-top:14px;">{top}</div>' if top else ""
    return (
        '<div class="fm-summary-head"><h3>Prediction</h3>'
        f'<span class="fm-badge">{confidence}</span></div>'
        f'<div class="fm-pred-name" style="margin-top:10px;">{html.escape(prediction.label)}</div>'
        f"{top_html}"
    )


def _single_weight_html(result) -> str:
    if result is None:
        return (
            '<div class="fm-summary-head"><h3>Portion Weight</h3></div>'
            '<div class="fm-note" style="margin-top:12px;">Weight appears after analysis.</div>'
        )

    weight = result.weight
    if weight is None or not weight.available:
        reason = "; ".join(weight.warnings) if weight and weight.warnings else "no weight source"
        return (
            '<div class="fm-summary-head"><h3>Portion Weight</h3>'
            '<span class="fm-badge">Unavailable</span></div>'
            f'<div class="fm-note" style="margin-top:12px;">{html.escape(reason)}</div>'
        )

    status = weight.status.capitalize()
    body = "".join(
        [
            _stat_html("Weight", _fmt_value(weight.grams, 1, " g"), "weight"),
            _stat_html("Source", html.escape(status), "dishes"),
        ]
    )
    return (
        '<div class="fm-summary-head"><h3>Portion Weight</h3>'
        f'<span class="fm-badge">{html.escape(status)}</span></div>'
        f'<div class="fm-stat-grid">{body}</div>'
        f'<div class="fm-note" style="margin-top:12px;">{html.escape(weight.method)}</div>'
    )


def _single_nutrition_html(result) -> str:
    if result is None:
        return (
            '<div class="fm-summary-head"><h3>Nutrition</h3></div>'
            '<div class="fm-note" style="margin-top:12px;">Nutrition appears after analysis.</div>'
        )

    nutrition = result.nutrition
    if nutrition is None or not nutrition.available:
        reason = nutrition.reason if nutrition else "not run"
        return (
            '<div class="fm-summary-head"><h3>Nutrition</h3>'
            '<span class="fm-badge">Unavailable</span></div>'
            f'<div class="fm-note" style="margin-top:12px;">{html.escape(str(reason))}</div>'
        )

    stats = "".join(
        _stat_html(
            label,
            _fmt_value(nutrition.nutrients.get(key), digits, f" {unit}"),
            icon,
        )
        for key, label, unit, digits, icon in _NUTRITION_META
        if key in nutrition.nutrients
    )
    basis = f"{nutrition.basis} · {_fmt_value(nutrition.grams, 0, ' g')}"
    return (
        '<div class="fm-summary-head"><h3>Nutrition</h3>'
        f'<span class="fm-badge">{html.escape(basis)}</span></div>'
        f'<div class="fm-stat-grid">{stats}</div>'
    )


_DISH_MACRO_META = [
    meta
    for meta in _NUTRITION_META
    if meta[0] in {"calories", "protein", "carbohydrates", "fats"}
]


def _per_dish_nutrition_html(result, reason=None) -> str:
    if result is None:
        note = html.escape(reason or "Analyze a plate to see nutrition for each dish.")
        return (
            '<div class="fm-summary-head"><h3>Nutrition by Dish</h3></div>'
            f'<div class="fm-note" style="margin-top:12px;">{note}</div>'
        )
    if not result.dishes:
        return (
            '<div class="fm-summary-head"><h3>Nutrition by Dish</h3></div>'
            '<div class="fm-note" style="margin-top:12px;">No dishes were detected.</div>'
        )

    included = list(result.included) if result.included else [True] * len(result.dishes)
    cards = []
    for dish, is_included in zip(result.dishes, included):
        prediction = dish.prediction
        if prediction is not None and prediction.available and prediction.label:
            name = prediction.label
        elif dish.detection is not None and dish.detection.label:
            name = dish.detection.label
        else:
            name = f"Food {dish.index}"

        weight = dish.weight
        if weight is not None and weight.available:
            weight_text = f"{_fmt_value(weight.grams, 0, ' g')} · {weight.status}"
        else:
            weight_text = "Weight unavailable"

        include_label = "Included" if is_included else "Excluded"
        include_class = "fm-badge" if is_included else "fm-badge fm-badge-muted"
        badges = (
            f'<span class="{include_class}">{include_label}</span>'
            f'<span class="fm-badge fm-badge-muted">{html.escape(weight_text)}</span>'
        )

        if dish.nutrition is not None and dish.nutrition.available:
            macros = "".join(
                '<div class="fm-dish-macro">'
                f"<span>{_icon(icon)}{html.escape(label)}</span>"
                f"<strong>{_fmt_value(dish.nutrition.nutrients.get(key), digits, f' {unit}')}</strong>"
                "</div>"
                for key, label, unit, digits, icon in _DISH_MACRO_META
                if key in dish.nutrition.nutrients
            )
            body = f'<div class="fm-dish-macros">{macros}</div>'
            if dish.nutrition.note:
                body += (
                    f'<div class="fm-note fm-dish-note">{html.escape(dish.nutrition.note)}</div>'
                )
        else:
            nutrition_reason = (
                dish.nutrition.reason if dish.nutrition is not None else "not run"
            )
            body = (
                '<div class="fm-note fm-dish-note">Nutrition unavailable: '
                f"{html.escape(str(nutrition_reason))}</div>"
            )

        excluded_class = "" if is_included else " fm-dish-excluded"
        cards.append(
            f'<article class="fm-dish-card{excluded_class}">'
            '<div class="fm-dish-head">'
            '<div>'
            f'<div class="fm-dish-name">{html.escape(str(name))}</div>'
            f'<div class="fm-note">Dish {dish.index}</div>'
            "</div>"
            f'<div class="fm-dish-badges">{badges}</div>'
            "</div>"
            f"{body}"
            "</article>"
        )

    dish_word = "dish" if len(result.dishes) == 1 else "dishes"
    return (
        '<div class="fm-summary-head">'
        f'<h3>Nutrition by Dish</h3><span class="fm-badge">{len(result.dishes)} {dish_word}</span>'
        "</div>"
        f'<div class="fm-dish-grid">{"".join(cards)}</div>'
    )


def _image_input(gr, label, height=None):
    """An image input with upload + camera, tolerant of older Gradio APIs."""
    kwargs = {"type": "filepath", "sources": ["upload", "webcam"], "label": label}
    if height is not None:
        kwargs["height"] = height
    try:
        return gr.Image(elem_classes="fm-image", **kwargs)
    except TypeError:
        kwargs.pop("height", None)
        return gr.Image(type="filepath", label=label)


def multi_detect(multi_pipeline, image, top_k, max_dishes, crop_padding):
    """Detect + classify a plate and return the UI values.

    Returns ``(annotated, summary_html, per_dish_html, report, payload, notes)``.
    """
    if image is None:
        reason = "No image provided. Upload or capture a plate to begin."
        return (
            None,
            _plate_summary_html(None, reason=reason),
            _per_dish_nutrition_html(None, reason=reason),
            "No image provided.",
            {},
            LIMITATIONS,
        )
    result = multi_pipeline.analyze(
        image,
        top_k=int(top_k),
        max_dishes=int(max_dishes),
        crop_padding=float(crop_padding),
    )
    if not result.available:
        reason = f"Analysis unavailable: {result.reason}"
        return (
            None,
            _plate_summary_html(None, reason=reason),
            _per_dish_nutrition_html(None, reason=reason),
            reason,
            {},
            LIMITATIONS,
        )

    detections = [d.detection for d in result.dishes if d.detection is not None]
    try:
        annotated = draw_detections(image, detections) if detections else image
    except Exception:
        annotated = image

    return (
        annotated,
        _plate_summary_html(result),
        _per_dish_nutrition_html(result),
        format_multi_analysis(result),
        multi_analysis_to_dict(result),
        LIMITATIONS,
    )


def _build_theme(gr):
    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.green,
        secondary_hue=gr.themes.colors.green,
        neutral_hue=gr.themes.colors.gray,
        radius_size=gr.themes.sizes.radius_lg,
        spacing_size=gr.themes.sizes.spacing_md,
    )
    return theme.set(
        body_background_fill="#f5f7f4",
        body_background_fill_dark="#f5f7f4",
        body_text_color="#1e293b",
        body_text_color_dark="#1e293b",
        body_text_color_subdued="#475569",
        body_text_color_subdued_dark="#475569",
        background_fill_primary="#ffffff",
        background_fill_primary_dark="#ffffff",
        background_fill_secondary="#f8faf9",
        background_fill_secondary_dark="#f8faf9",
        block_background_fill="#ffffff",
        block_background_fill_dark="#ffffff",
        block_border_color="#e2e8f0",
        block_border_color_dark="#e2e8f0",
        block_label_text_color="#0f172a",
        block_label_text_color_dark="#0f172a",
        block_info_text_color="#475569",
        block_info_text_color_dark="#475569",
        block_shadow="0 12px 34px rgba(15, 23, 42, 0.05)",
        block_shadow_dark="0 12px 34px rgba(15, 23, 42, 0.05)",
        block_radius="20px",
        button_primary_background_fill="#14532d",
        button_primary_background_fill_dark="#14532d",
        button_primary_background_fill_hover="#166534",
        button_primary_background_fill_hover_dark="#166534",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        button_primary_border_color="#14532d",
        button_primary_border_color_dark="#14532d",
        button_secondary_text_color="#0f172a",
        button_secondary_text_color_dark="#0f172a",
        color_accent="#14532d",
        color_accent_soft="#dcfce7",
        color_accent_soft_dark="#dcfce7",
        input_background_fill="#ffffff",
        input_background_fill_dark="#ffffff",
        input_border_color="#dbe3dc",
        input_border_color_dark="#dbe3dc",
        input_placeholder_color="#64748b",
        input_placeholder_color_dark="#64748b",
        slider_color="#14532d",
        slider_color_dark="#14532d",
        table_text_color="#0f172a",
        table_text_color_dark="#0f172a",
        table_border_color="#e2e8f0",
        table_border_color_dark="#e2e8f0",
        table_even_background_fill="#ffffff",
        table_even_background_fill_dark="#ffffff",
        table_odd_background_fill="#f8faf9",
        table_odd_background_fill_dark="#f8faf9",
        accordion_text_color="#0f172a",
        accordion_text_color_dark="#0f172a",
        checkbox_label_text_color="#0f172a",
        checkbox_label_text_color_dark="#0f172a",
    )


def build_interface(pipeline: FoodAnalysisPipeline, multi_pipeline=None):
    import gradio as gr

    def run(image, measured, top_k):
        if image is None:
            return (
                _single_prediction_html(None),
                _single_weight_html(None),
                _single_nutrition_html(None),
                "No image provided.",
                "",
                "",
                "",
                {},
                LIMITATIONS,
            )
        result = pipeline.analyze(
            image, measured_grams=_clean_measured(measured), top_k=int(top_k)
        )
        notes = "\n".join(f"- {w}" for w in result.warnings) or "No warnings."
        return (
            _single_prediction_html(result),
            _single_weight_html(result),
            _single_nutrition_html(result),
            _classification_text(result),
            result.detection.summary() if result.detection else "not run",
            result.weight.summary(),
            _macros_text(result),
            analysis_to_dict(result),
            notes + "\n\n" + LIMITATIONS,
        )

    def run_multi_detect(image, top_k, max_dishes, crop_padding):
        return multi_detect(multi_pipeline, image, top_k, max_dishes, crop_padding)

    def switch_view(choice):
        return (
            gr.update(visible=choice == "Single Dish"),
            gr.update(visible=choice == "Multi-Dish"),
            gr.update(visible=choice == "About"),
        )

    theme = _build_theme(gr)
    with gr.Blocks(title="FoodMacros") as demo:
        with gr.Row(elem_classes="fm-nav"):
            with gr.Column(scale=3, min_width=260):
                gr.HTML(BRAND_HTML)
            with gr.Column(scale=2, min_width=280):
                nav = gr.Radio(
                    choices=["Single Dish", "Multi-Dish", "About"],
                    value="Single Dish",
                    show_label=False,
                    container=False,
                    elem_classes="fm-nav-radio",
                )

        with gr.Column(visible=True) as single_view:
            _single_dish_tab(pipeline, run, gr)

        with gr.Column(visible=False) as multi_view:
            if multi_pipeline is not None:
                _multi_dish_tab(multi_pipeline, run_multi_detect, gr)
            else:
                with gr.Column(elem_classes="fm-card"):
                    gr.Markdown(
                        "Multi-dish analysis is unavailable because no multi-dish "
                        "pipeline was provided."
                    )

        with gr.Column(visible=False) as about_view:
            _about_tab(gr)

        nav.change(
            switch_view,
            inputs=nav,
            outputs=[single_view, multi_view, about_view],
        )
    demo.foodmacros_theme = theme
    demo.foodmacros_css = CSS
    return demo


def _single_dish_tab(pipeline: FoodAnalysisPipeline, run, gr):
    with gr.Row(equal_height=False, elem_classes="fm-layout-row"):
        with gr.Column(scale=1, min_width=300, elem_classes="fm-card"):
            gr.HTML(
                _section_html(
                    "Single Dish",
                    "Upload or capture one dish, then let FoodMacros estimate its "
                    "weight and nutrition.",
                )
            )
            image_input = _image_input(gr, "Food image", height=300)
            measured_input = gr.Number(
                value=None,
                label="Measured weight (g, optional)",
                precision=1,
                info="Enter a measured weight to override the automatic estimate.",
            )
            top_k = gr.Slider(
                1,
                5,
                value=int(pipeline.config.ui.max_top_k),
                step=1,
                label="Top-k predictions",
            )
            analyze_button = gr.Button(
                "Analyze Dish", variant="primary", elem_classes="fm-primary"
            )

        with gr.Column(scale=2):
            with gr.Column(elem_classes="fm-card fm-summary"):
                prediction_out = gr.HTML(_single_prediction_html(None))
            with gr.Row(equal_height=False, elem_classes="fm-layout-row"):
                with gr.Column(scale=1, elem_classes="fm-card"):
                    weight_out_html = gr.HTML(_single_weight_html(None))
                with gr.Column(scale=1, elem_classes="fm-card"):
                    nutrition_out = gr.HTML(_single_nutrition_html(None))
            with gr.Accordion("Detailed analysis", open=False):
                cls_out = gr.Textbox(label="Classification", lines=4)
                det_out = gr.Textbox(label="Detection", lines=2)
                weight_out = gr.Textbox(label="Portion weight", lines=2)
                macros_out = gr.Textbox(label="Macronutrients", lines=8)
                json_out = gr.JSON(label="Structured result")
                notes_out = gr.Markdown(value=LIMITATIONS, elem_classes="fm-limitations")

    analyze_button.click(
        run,
        inputs=[image_input, measured_input, top_k],
        outputs=[
            prediction_out,
            weight_out_html,
            nutrition_out,
            cls_out,
            det_out,
            weight_out,
            macros_out,
            json_out,
            notes_out,
        ],
    )


def _multi_dish_tab(multi_pipeline: MultiDishPipeline, run_multi_detect, gr):
    det_cfg = multi_pipeline.config.detection

    with gr.Row(equal_height=False, elem_classes="fm-layout-row"):
        with gr.Column(scale=1, min_width=300, elem_classes="fm-card"):
            gr.HTML(
                _section_html(
                    "Multi-Dish Analysis",
                    "Detect every dish on the plate and review the plate summary "
                    "and nutrition for each dish.",
                )
            )
            top_k = gr.Slider(
                1,
                5,
                value=int(multi_pipeline.config.ui.max_top_k),
                step=1,
                label="Top-k per dish",
            )
            max_dishes = gr.Slider(
                1,
                10,
                value=int(det_cfg.max_dishes),
                step=1,
                label="Max dishes",
            )
            with gr.Accordion("Advanced Settings", open=False):
                crop_padding = gr.Slider(
                    0.0,
                    0.3,
                    value=float(det_cfg.crop_padding),
                    step=0.01,
                    label="Extra Area Around Food",
                    info="Controls how much surrounding area is included when analyzing each dish.",
                )
            detect_button = gr.Button(
                "Analyze Plate", variant="primary", elem_classes="fm-primary"
            )

        with gr.Column(scale=3):
            with gr.Row(equal_height=True, elem_classes="fm-image-row"):
                with gr.Column(scale=1, min_width=280, elem_classes="fm-card"):
                    gr.HTML(_section_html("Input Image"))
                    image_input = _image_input(gr, "Upload or capture", height=340)
                with gr.Column(scale=1, min_width=280, elem_classes="fm-card"):
                    gr.HTML(_section_html("Detected Dishes"))
                    annotated_out = gr.Image(
                        label="Detected dishes",
                        interactive=False,
                        height=340,
                        elem_classes="fm-image",
                    )

        with gr.Column(scale=1, min_width=280, elem_classes="fm-card fm-summary"):
            summary_out = gr.HTML(_plate_summary_html(None))

    with gr.Column(elem_classes="fm-card"):
        per_dish_out = gr.HTML(_per_dish_nutrition_html(None))

    with gr.Accordion("Technical details", open=False):
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
                "About FoodMacros",
                "FoodMacros combines food classification, detection, portion-weight "
                "estimation, and nutrition lookup in one local dashboard.",
            )
        )
        gr.Markdown(
            """
### How it works

1. **Classify** the dish from the uploaded image.
2. **Detect** individual food items on a plate.
3. **Estimate weight** with a validated regression model, or use your measured value.
4. **Look up nutrition** and aggregate the included dishes into a plate total.

### Reliability

Weight is only reported as *measured*, *estimated*, or *experimental*. When no
reliable weight source is available, macros are shown for a reference quantity
only. Classification confidence is not a guarantee of correctness.
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

    multi_pipeline = create_multi_pipeline(config, args.model_dir, args.regressor_dir)
    for note in multi_pipeline.notes:
        print(f"note: {note}")

    import gradio as gr

    demo = build_interface(pipeline, multi_pipeline=multi_pipeline)
    launch_kwargs = {
        "share": config.ui.share,
        "server_port": config.ui.server_port,
        "theme": getattr(demo, "foodmacros_theme", _build_theme(gr)),
        "css": getattr(demo, "foodmacros_css", CSS),
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
