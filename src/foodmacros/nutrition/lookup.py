"""Label -> nutrition lookup with an explicit, documented multi-row rule.

The source CSV contains several rows per dish at different weights. Rather
than silently taking the first row, a label's per-100 g density is derived by
one of these documented strategies (default ``nearest_weight``):

* ``nearest_weight`` — use the row whose weight is closest to the reference
  weight (default 100 g) and scale it proportionally to 100 g.
* ``per_gram_mean`` — average ``value / weight * 100`` across all rows
  (slope of the proportional model; robust to sampling noise).
* ``interpolate`` — linearly interpolate each nutrient's *per-gram rate*
  (value / weight) at the reference weight, then express it per 100 g.
  Values outside the observed weight range are clamped to the nearest row.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .csv_schema import (
    LABEL_COLUMN,
    REQUIRED_COLUMNS,
    WEIGHT_COLUMN,
    SchemaReport,
    load_frame,
)
from .portion import NUTRIENT_COLUMNS, per_100g_scale, round_nutrients, scale_to_grams

VALID_STRATEGIES = ("nearest_weight", "per_gram_mean", "interpolate")


def normalize_label(label) -> str:
    """Lower-case a label and collapse any run of non-alphanumerics to ``_``.

    Used only as a *fallback* match so that e.g. ``"Fish and Chips"`` and
    ``"fish_and_chips"`` resolve to the same row. Exact and case-insensitive
    matches always take priority.
    """
    return re.sub(r"[^a-z0-9]+", "_", str(label).strip().lower()).strip("_")


# Alternative spellings that do not normalise to the CSV label. Maps a model
# label (as it appears in ``class_names.json``) to the nutrition-row label.
# Only three Food-101 labels genuinely need this; the rest are covered by
# :func:`normalize_label` (e.g. "Fish and Chips" -> "fish_and_chips").
LABEL_ALIASES = {
    "cup_cakes": "Cupcakes",
    "grilled_cheese_sandwich": "Grilled Cheese",
    "lobster_roll_sandwich": "Lobster Roll",
}


@dataclass
class PortionResult:
    label: str
    grams: float
    strategy: str
    per_100g: Dict[str, float]
    nutrients: Dict[str, float]
    reference_weight: float = 100.0
    note: str = ""

    def as_display(self, ndigits: int = 1) -> Dict[str, object]:
        """Rounded, presentation-ready view (missing values -> ``None``)."""
        return {
            "label": self.label,
            "grams": round(self.grams, ndigits),
            "strategy": self.strategy,
            "per_100g": round_nutrients(self.per_100g, ndigits),
            "nutrients": round_nutrients(self.nutrients, ndigits),
            "note": self.note,
        }


class NutritionTable:
    """In-memory nutrition index built from a validated CSV frame."""

    def __init__(
        self,
        frame: pd.DataFrame,
        schema: Optional[SchemaReport] = None,
        dropped_rows: int = 0,
        source: Optional[str] = None,
    ) -> None:
        self.frame = frame
        self.schema = schema
        self.dropped_rows = int(dropped_rows)
        self.source = source
        self._nutrient_columns = [c for c in NUTRIENT_COLUMNS if c in frame.columns]
        self._groups: Dict[str, pd.DataFrame] = {
            str(label): group.reset_index(drop=True)
            for label, group in frame.groupby(LABEL_COLUMN, sort=True)
        }
        self._lower_map = {label.lower(): label for label in self._groups}
        # Normalised fallback index: handles spacing/underscore/hyphen/spelling
        # differences (e.g. "Fish and Chips" -> "fish_and_chips"). Exact and
        # case-insensitive matches still win because they are tried first.
        self._norm_map: Dict[str, str] = {}
        for label in self._groups:
            self._norm_map.setdefault(normalize_label(label), label)
        for alias, target in LABEL_ALIASES.items():
            if target in self._groups:
                self._norm_map[normalize_label(alias)] = target

    # -- construction -----------------------------------------------------
    @classmethod
    def from_csv(
        cls, path: str | Path, required=REQUIRED_COLUMNS
    ) -> "NutritionTable":
        frame, report, dropped = load_frame(path, required=required)
        return cls(frame, schema=report, dropped_rows=dropped, source=str(path))

    # -- inspection -------------------------------------------------------
    @property
    def labels(self) -> List[str]:
        return sorted(self._groups)

    @property
    def available_nutrients(self) -> List[str]:
        return list(self._nutrient_columns)

    def _canonical(self, label) -> Optional[str]:
        key = str(label).strip()
        if key in self._groups:
            return key
        low = self._lower_map.get(key.lower())
        if low is not None:
            return low
        return self._norm_map.get(normalize_label(key))

    def has_label(self, label) -> bool:
        return self._canonical(label) is not None

    def rows_for(self, label) -> Optional[pd.DataFrame]:
        canonical = self._canonical(label)
        if canonical is None:
            return None
        return self._groups[canonical]

    def missing_labels(self, labels) -> List[str]:
        return [str(l) for l in labels if not self.has_label(l)]

    def extra_labels(self, labels) -> List[str]:
        known = {str(l).lower() for l in labels}
        return [l for l in self.labels if l.lower() not in known]

    # -- internals --------------------------------------------------------
    def _weights(self, gdf: pd.DataFrame) -> np.ndarray:
        return gdf[WEIGHT_COLUMN].to_numpy(dtype=float)

    def _row_nutrients(self, row: pd.Series) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for column in self._nutrient_columns:
            value = row[column] if column in row else float("nan")
            out[column] = float("nan") if pd.isna(value) else float(value)
        return out

    def _nearest_row_index(self, weights: np.ndarray, target: float) -> int:
        return int(np.argmin(np.abs(weights - target)))

    def _per100_nearest(self, gdf: pd.DataFrame, target: float) -> tuple[Dict[str, float], float]:
        weights = self._weights(gdf)
        idx = self._nearest_row_index(weights, target)
        basis_weight = float(weights[idx])
        nutrients = self._row_nutrients(gdf.iloc[idx])
        return per_100g_scale(nutrients, basis_weight), basis_weight

    def _per100_per_gram_mean(self, gdf: pd.DataFrame) -> Dict[str, float]:
        weights = self._weights(gdf)
        out: Dict[str, float] = {}
        for column in self._nutrient_columns:
            values = gdf[column].to_numpy(dtype=float) if column in gdf else np.full(len(gdf), np.nan)
            mask = ~np.isnan(values)
            if mask.any():
                out[column] = float(np.mean(values[mask] / weights[mask] * 100.0))
            else:
                out[column] = float("nan")
        return out

    def _per100_interpolate(self, gdf: pd.DataFrame, target: float) -> Dict[str, float]:
        weights = self._weights(gdf)
        out: Dict[str, float] = {}
        for column in self._nutrient_columns:
            values = gdf[column].to_numpy(dtype=float) if column in gdf else np.full(len(gdf), np.nan)
            mask = ~np.isnan(values)
            if not mask.any():
                out[column] = float("nan")
                continue
            w = weights[mask]
            v = values[mask]
            rates = v / w  # per-gram density (weights are guaranteed positive)
            if mask.sum() == 1:
                out[column] = float(rates[0] * 100.0)
                continue
            order = np.argsort(w)
            rate = float(np.interp(target, w[order], rates[order]))
            out[column] = rate * 100.0
        return out

    # -- public API -------------------------------------------------------
    def per_100g(
        self,
        label,
        strategy: str = "nearest_weight",
        reference_weight: float = 100.0,
    ) -> Dict[str, float]:
        """Return nutrient density per 100 g for ``label``."""
        if strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Valid: {list(VALID_STRATEGIES)}."
            )
        canonical = self._canonical(label)
        if canonical is None:
            sample = ", ".join(self.labels[:5])
            raise KeyError(
                f"Label '{label}' not found in nutrition table "
                f"({len(self._groups)} labels available, e.g. {sample})."
            )
        gdf = self._groups[canonical]
        if strategy == "nearest_weight":
            return self._per100_nearest(gdf, reference_weight)[0]
        if strategy == "per_gram_mean":
            return self._per100_per_gram_mean(gdf)
        return self._per100_interpolate(gdf, reference_weight)

    def portion(
        self,
        label,
        grams: float,
        strategy: str = "nearest_weight",
        reference_weight: float = 100.0,
    ) -> PortionResult:
        """Compute nutrients for ``grams`` grams of ``label``."""
        if grams is None:
            raise ValueError("grams must not be None.")
        if grams < 0:
            raise ValueError("grams must be non-negative.")
        per_100g = self.per_100g(label, strategy=strategy, reference_weight=reference_weight)
        nutrients = scale_to_grams(per_100g, grams)
        canonical = self._canonical(label)
        note = (
            f"per-100 g derived via '{strategy}' "
            f"(reference {reference_weight:g} g; {len(self._groups[canonical])} source row(s))."
        )
        return PortionResult(
            label=canonical,
            grams=float(grams),
            strategy=strategy,
            per_100g=per_100g,
            nutrients=nutrients,
            reference_weight=float(reference_weight),
            note=note,
        )
