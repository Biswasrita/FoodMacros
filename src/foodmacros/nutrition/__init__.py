"""Nutrition layer: schema validation, label lookup and portion arithmetic."""

from .csv_schema import (
    LABEL_COLUMN,
    NUTRIENT_COLUMNS,
    REQUIRED_COLUMNS,
    WEIGHT_COLUMN,
    NutritionSchemaError,
    SchemaReport,
    load_frame,
    validate_schema,
)
from .lookup import VALID_STRATEGIES, NutritionTable, PortionResult
from .portion import per_100g_scale, round_nutrients, scale_to_grams

__all__ = [
    "LABEL_COLUMN",
    "NUTRIENT_COLUMNS",
    "REQUIRED_COLUMNS",
    "WEIGHT_COLUMN",
    "NutritionSchemaError",
    "SchemaReport",
    "load_frame",
    "validate_schema",
    "VALID_STRATEGIES",
    "NutritionTable",
    "PortionResult",
    "per_100g_scale",
    "round_nutrients",
    "scale_to_grams",
]
