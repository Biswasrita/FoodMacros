"""Formatting of multi-dish analysis results for humans and machines.

Mirrors :mod:`foodmacros.reporting` but for a plate with several dishes. All
numbers come from the pipeline; missing values render as ``None``/``n/a``.
"""

from __future__ import annotations

from typing import Dict, Optional

from .nutrition.portion import round_nutrients
from .reporting import LIMITATIONS, NUTRIENT_LABELS, format_nutrients


def _fmt_value(value: Optional[float], ndigits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{round(float(value), ndigits):g}"


def format_multi_analysis(result) -> str:
    lines = ["=== Multi-Dish Food Analysis ==="]

    if not result.available:
        lines.append(f"Analysis      : unavailable ({result.reason})")
        lines.append(LIMITATIONS)
        return "\n".join(lines)

    lines.append(f"Detection     : {result.detection.summary() if result.detection else 'not run'}")
    lines.append(f"Dishes        : {len(result.dishes)}")

    included = list(result.included) if result.included else [True] * len(result.dishes)
    for dish, inc in zip(result.dishes, included):
        lines.append("")
        mark = "x" if inc else " "
        lines.append(f"-- [{mark}] {dish.summary()}")
        lines.append(f"   box         : {tuple(round(float(v), 1) for v in dish.box) if dish.box else 'whole image'}")
        lines.append(f"   weight      : {dish.weight.summary() if dish.weight else 'not run'}")
        if dish.prediction is not None and dish.prediction.available:
            top = ", ".join(f"{n} {p:.1%}" for n, p in dish.prediction.top_k)
            lines.append(f"   top-{len(dish.prediction.top_k):<2}      : {top}")
        if dish.nutrition is not None and dish.nutrition.available:
            lines.append(f"   macros ({dish.nutrition.basis}, {dish.nutrition.grams:g} g):")
            lines.append(format_nutrients(dish.nutrition.nutrients))
        else:
            reason = dish.nutrition.reason if dish.nutrition else "not run"
            lines.append(f"   macros      : unavailable ({reason})")

    lines.append("")
    if result.totals_basis == "none":
        reason = result.totals_reason or "no dish had usable nutrition data"
        lines.append(f"Plate totals  : unavailable ({reason})")
    else:
        header = f"Plate totals ({result.totals_basis}"
        if result.total_grams is not None:
            header += f", {result.total_grams:g} g"
        header += "):"
        lines.append(header)
        lines.append(format_nutrients(result.totals))

    if result.warnings:
        lines.append("Warnings:")
        lines.extend(f"  - {w}" for w in result.warnings)
    if result.notes:
        lines.append("Notes:")
        lines.extend(f"  - {n}" for n in result.notes)
    lines.append(LIMITATIONS)
    return "\n".join(lines)


def multi_analysis_to_dict(result) -> Dict[str, object]:
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

    dishes = []
    included_flags = list(result.included) if result.included else [True] * len(result.dishes)
    for pos, dish in enumerate(result.dishes):
        prediction = None
        if dish.prediction is not None:
            prediction = {
                "available": dish.prediction.available,
                "label": dish.prediction.label,
                "confidence": dish.prediction.confidence,
                "top_k": dish.prediction.top_k,
                "reason": dish.prediction.reason,
            }
        nutrition = None
        if dish.nutrition is not None:
            nutrition = {
                "available": dish.nutrition.available,
                "label": dish.nutrition.label,
                "basis": dish.nutrition.basis,
                "grams": dish.nutrition.grams,
                "nutrients": (
                    round_nutrients(dish.nutrition.nutrients)
                    if dish.nutrition.available
                    else None
                ),
                "reason": dish.nutrition.reason,
                "note": dish.nutrition.note,
            }
        weight = None
        if dish.weight is not None:
            weight = {
                "status": dish.weight.status,
                "grams": dish.weight.grams,
                "method": dish.weight.method,
                "confidence": dish.weight.confidence,
            }
        dishes.append(
            {
                "index": dish.index,
                "included": bool(included_flags[pos]) if pos < len(included_flags) else True,
                "detection": (
                    {"label": dish.detection.label, "confidence": dish.detection.confidence}
                    if dish.detection is not None
                    else None
                ),
                "box": tuple(dish.box),
                "classification": prediction,
                "weight": weight,
                "nutrition": nutrition,
                "warnings": list(dish.warnings),
                "notes": list(dish.notes),
            }
        )

    return {
        "available": result.available,
        "reason": result.reason,
        "detection": detection,
        "dishes": dishes,
        "totals": (
            round_nutrients(result.totals) if result.totals_basis != "none" else None
        ),
        "totals_basis": result.totals_basis,
        "totals_reason": result.totals_reason,
        "total_grams": result.total_grams,
        "reference_quantity_g": result.reference_quantity_g,
        "warnings": list(result.warnings),
        "notes": list(result.notes),
        "limitations": LIMITATIONS,
    }
