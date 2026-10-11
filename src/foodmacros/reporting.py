"""Formatting of analysis results for humans and machines.

All numbers shown are produced by the pipeline; nothing is fabricated. Missing
values render as ``None``/``n/a`` rather than 0.
"""

from __future__ import annotations

from typing import Dict, Optional

from .nutrition.portion import round_nutrients

NUTRIENT_LABELS = {
    "calories": "Calories (kcal)",
    "protein": "Protein (g)",
    "carbohydrates": "Carbohydrates (g)",
    "fats": "Fat (g)",
    "fiber": "Fiber (g)",
    "sugars": "Sugars (g)",
    "sodium": "Sodium (mg)",
}

LIMITATIONS = (
    "Limitations: weight is only reported as 'measured' (user-supplied), "
    "'estimated' (validated Nutrition5k regression) or 'experimental' "
    "(depth x approximate density). When no such source is available the weight "
    "is 'unavailable' and macros are shown for a reference quantity only. "
    "Classification confidence is not a guarantee of correctness."
)


def _fmt_value(value: Optional[float], ndigits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{round(float(value), ndigits):g}"


def format_nutrients(nutrients: Dict[str, Optional[float]], ndigits: int = 1) -> str:
    rounded = round_nutrients(nutrients, ndigits)
    lines = []
    for key in NUTRIENT_LABELS:
        if key in rounded:
            lines.append(f"  {NUTRIENT_LABELS[key]}: {_fmt_value(rounded[key], ndigits)}")
    for key, value in rounded.items():
        if key not in NUTRIENT_LABELS:
            lines.append(f"  {key}: {_fmt_value(value, ndigits)}")
    return "\n".join(lines) if lines else "  (no nutrients)"


def format_analysis(result) -> str:
    lines = ["=== Food Analysis ==="]

    if result.prediction is None or not result.prediction.available:
        reason = result.prediction.reason if result.prediction else "not run"
        lines.append(f"Classification : unavailable ({reason})")
    else:
        top = ", ".join(f"{n} {p:.1%}" for n, p in result.prediction.top_k)
        lines.append(f"Classification : {result.prediction.label} ({result.prediction.confidence:.1%})")
        lines.append(f"  top-{len(result.prediction.top_k)}: {top}")

    lines.append(f"Detection      : {result.detection.summary() if result.detection else 'not run'}")
    lines.append(f"Weight         : {result.weight.summary()}")

    if result.nutrition.available:
        lines.append(f"Macros ({result.nutrition.basis}, {result.nutrition.grams:g} g):")
        lines.append(format_nutrients(result.nutrition.nutrients))
        if result.nutrition.note:
            lines.append(f"  note: {result.nutrition.note}")
    else:
        lines.append(f"Macros         : unavailable ({result.nutrition.reason})")

    if result.warnings:
        lines.append("Warnings:")
        lines.extend(f"  - {w}" for w in result.warnings)
    if result.notes:
        lines.append("Notes:")
        lines.extend(f"  - {n}" for n in result.notes)
    lines.append(LIMITATIONS)
    return "\n".join(lines)


def analysis_to_dict(result) -> Dict[str, object]:
    prediction = None
    if result.prediction is not None:
        prediction = {
            "available": result.prediction.available,
            "label": result.prediction.label,
            "confidence": result.prediction.confidence,
            "top_k": result.prediction.top_k,
            "reason": result.prediction.reason,
        }
    detection = None
    if result.detection is not None:
        detection = {
            "available": result.detection.available,
            "detections": [
                {"label": d.label, "confidence": d.confidence, "box": d.box_xyxy}
                for d in result.detection.detections
            ],
            "reason": result.detection.reason,
        }
    nutrition = {
        "available": result.nutrition.available,
        "label": result.nutrition.label,
        "basis": result.nutrition.basis,
        "grams": result.nutrition.grams,
        "nutrients": round_nutrients(result.nutrition.nutrients) if result.nutrition.available else None,
        "per_100g": round_nutrients(result.nutrition.per_100g) if result.nutrition.available else None,
        "reason": result.nutrition.reason,
        "note": result.nutrition.note,
    }
    return {
        "classification": prediction,
        "detection": detection,
        "weight": {
            "status": result.weight.status,
            "grams": result.weight.grams,
            "method": result.weight.method,
            "confidence": result.weight.confidence,
        },
        "nutrition": nutrition,
        "reference_quantity_g": result.reference_quantity_g,
        "warnings": list(result.warnings),
        "notes": list(result.notes),
        "limitations": LIMITATIONS,
    }
