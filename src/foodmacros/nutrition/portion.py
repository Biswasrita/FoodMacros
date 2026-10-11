"""Portion arithmetic for macronutrients.

The canonical quantity is a nutrient density *per 100 g* (derived from the
source data), which is then scaled to an actual portion weight in grams.
Nothing here assumes a fixed serving size.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping

NUTRIENT_COLUMNS = [
    "calories",
    "protein",
    "carbohydrates",
    "fats",
    "fiber",
    "sugars",
    "sodium",
]


def _is_missing(value) -> bool:
    """True for ``None``/``NaN``/non-numeric values."""
    try:
        return value is None or math.isnan(float(value))
    except (TypeError, ValueError):
        return True


def per_100g_scale(nutrients: Mapping[str, float], basis_weight: float) -> Dict[str, float]:
    """Convert nutrient amounts measured at ``basis_weight`` grams to per-100 g.

    Nutrients are assumed to scale proportionally with mass.
    """
    if basis_weight is None or basis_weight <= 0:
        raise ValueError("basis_weight must be a positive number of grams.")
    out: Dict[str, float] = {}
    for name, value in nutrients.items():
        out[name] = float("nan") if _is_missing(value) else float(value) / basis_weight * 100.0
    return out


def scale_to_grams(per_100g: Mapping[str, float], grams: float) -> Dict[str, float]:
    """Scale a per-100 g nutrient dictionary to ``grams`` grams."""
    if grams is None or grams < 0:
        raise ValueError("grams must be a non-negative number.")
    out: Dict[str, float] = {}
    for name, value in per_100g.items():
        out[name] = float("nan") if _is_missing(value) else float(value) * grams / 100.0
    return out


def round_nutrients(nutrients: Mapping[str, float], ndigits: int = 1) -> Dict[str, float | None]:
    """Round values for display; missing values become ``None`` (not 0)."""
    out: Dict[str, float | None] = {}
    for name, value in nutrients.items():
        out[name] = None if _is_missing(value) else round(float(value), ndigits)
    return out
