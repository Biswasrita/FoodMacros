"""Nutrition CSV schema validation and safe loading.

Column names are validated at runtime (no assumptions), numeric columns are
coerced, and invalid rows are dropped with a count so the user can see exactly
what was discarded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

import pandas as pd

from .portion import NUTRIENT_COLUMNS

LABEL_COLUMN = "label"
WEIGHT_COLUMN = "weight"

# Minimal set required to compute a portion's macros.
REQUIRED_COLUMNS = ["label", "weight", "calories", "protein", "carbohydrates", "fats"]


class NutritionSchemaError(Exception):
    """Raised when a CSV does not contain the required nutrition columns."""


@dataclass
class SchemaReport:
    columns: List[str]
    required_present: List[str] = field(default_factory=list)
    optional_present: List[str] = field(default_factory=list)
    missing_required: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.missing_required

    def summary(self) -> str:
        return (
            f"columns={len(self.columns)} required_ok={self.required_present} "
            f"optional={self.optional_present} missing={self.missing_required}"
        )


def validate_schema(
    columns: Sequence[str],
    required: Sequence[str] = REQUIRED_COLUMNS,
    nutrient_columns: Sequence[str] = NUTRIENT_COLUMNS,
) -> SchemaReport:
    """Check for required columns, case-insensitively, without modifying data."""
    lower = {str(c).strip().lower() for c in columns}
    required_lower = [c.lower() for c in required]
    present = [c for c in required_lower if c in lower]
    missing = [c for c in required_lower if c not in lower]
    optional = [c for c in nutrient_columns if c in lower and c not in required_lower]
    return SchemaReport(
        columns=list(columns),
        required_present=present,
        optional_present=optional,
        missing_required=missing,
    )


def load_frame(
    path: str | Path,
    required: Sequence[str] = REQUIRED_COLUMNS,
) -> tuple[pd.DataFrame, SchemaReport, int]:
    """Load and sanitise a nutrition CSV.

    Returns ``(frame, schema_report, dropped_rows)``. Column names are
    lower-cased/stripped, numeric columns are coerced, and rows with a missing
    label or a non-positive weight are dropped.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Nutrition CSV not found: {path}")

    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # malformed CSV should surface clearly
        raise NutritionSchemaError(f"Could not read nutrition CSV '{path}': {exc}") from exc

    frame.columns = [str(c).strip().lower() for c in frame.columns]
    report = validate_schema(frame.columns, required=required)
    if not report.is_valid:
        raise NutritionSchemaError(
            f"Nutrition CSV '{path}' is missing required column(s): "
            f"{report.missing_required}. Present columns: {list(frame.columns)}."
        )

    frame[LABEL_COLUMN] = (
        frame[LABEL_COLUMN].astype("string").fillna("").str.strip()
    )
    frame[WEIGHT_COLUMN] = pd.to_numeric(frame[WEIGHT_COLUMN], errors="coerce")
    for column in NUTRIENT_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    before = len(frame)
    valid_label = frame[LABEL_COLUMN].str.len() > 0
    valid_weight = frame[WEIGHT_COLUMN].notna() & (frame[WEIGHT_COLUMN] > 0)
    frame = frame[valid_label & valid_weight].reset_index(drop=True)
    frame[LABEL_COLUMN] = frame[LABEL_COLUMN].astype(str)
    dropped = before - len(frame)
    return frame, report, dropped
