"""Stage 3 tests for nutrition schema validation and label lookup."""

import math

import pytest

from foodmacros.nutrition.csv_schema import (
    NutritionSchemaError,
    load_frame,
    validate_schema,
)
from foodmacros.nutrition.lookup import NutritionTable

HEADER = "label,weight,calories,protein,carbohydrates,fats,fiber,sugars,sodium"
ROWS = [
    "apple_pie,100,237,2.4,34,11,1.6,15,266",
    "apple_pie,200,474,4.8,68,22,3.2,30,532",
    "solo,150,300,6,40,10,2,5,100",
    "partial,100,200,5,30,8,,,",
    ",100,50,1,1,1,0,0,0",       # empty label -> dropped
    "badweight,abc,50,1,1,1,0,0,0",  # non-numeric weight -> dropped
]


@pytest.fixture
def precise_csv(tmp_path):
    path = tmp_path / "nutrition.csv"
    path.write_text("\n".join([HEADER] + ROWS), encoding="utf-8")
    return path


@pytest.fixture
def table(precise_csv):
    return NutritionTable.from_csv(precise_csv)


def test_validate_schema_accepts_required_columns():
    report = validate_schema(HEADER.split(","))
    assert report.is_valid
    assert set(report.required_present) == {
        "label", "weight", "calories", "protein", "carbohydrates", "fats"
    }
    assert set(report.optional_present) == {"fiber", "sugars", "sodium"}


def test_validate_schema_flags_missing_required():
    report = validate_schema(["label", "weight", "calories"])
    assert not report.is_valid
    assert "protein" in report.missing_required


def test_load_frame_raises_on_missing_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("label,weight\napple,100\n", encoding="utf-8")
    with pytest.raises(NutritionSchemaError):
        load_frame(bad)


def test_load_frame_drops_invalid_rows(precise_csv):
    frame, report, dropped = load_frame(precise_csv)
    assert report.is_valid
    assert dropped == 2
    assert len(frame) == 4


def test_load_frame_is_case_insensitive(tmp_path):
    path = tmp_path / "upper.csv"
    path.write_text("Label,Weight,Calories,Protein,Carbohydrates,Fats\nX,100,10,1,2,3\n", encoding="utf-8")
    frame, report, _ = load_frame(path)
    assert report.is_valid
    assert list(frame.columns[:2]) == ["label", "weight"]


def test_table_labels(table):
    assert table.labels == ["apple_pie", "partial", "solo"]


def test_lookup_is_case_insensitive(table):
    assert table.has_label("APPLE_PIE")
    assert table.has_label("apple_pie")
    assert not table.has_label("nope")


def test_missing_labels(table):
    assert table.missing_labels(["apple_pie", "nope"]) == ["nope"]


def test_per_100g_nearest_weight(table):
    per = table.per_100g("apple_pie", strategy="nearest_weight", reference_weight=100.0)
    assert per["calories"] == pytest.approx(237.0)
    assert per["protein"] == pytest.approx(2.4)


def test_per_100g_nearest_scales_single_row(table):
    # Only one row (150 g, 300 kcal) -> per-100 g must be 200, not 300.
    per = table.per_100g("solo", strategy="nearest_weight")
    assert per["calories"] == pytest.approx(200.0)


def test_per_100g_per_gram_mean(table):
    per = table.per_100g("apple_pie", strategy="per_gram_mean")
    assert per["calories"] == pytest.approx(237.0)


def test_per_100g_interpolate_two_rows(table):
    per = table.per_100g("apple_pie", strategy="interpolate", reference_weight=100.0)
    assert per["calories"] == pytest.approx(237.0)


def test_per_100g_interpolate_single_row_is_density(table):
    per = table.per_100g("solo", strategy="interpolate")
    assert per["calories"] == pytest.approx(200.0)


def test_partial_row_keeps_missing_nutrients_missing(table):
    per = table.per_100g("partial")
    assert per["calories"] == pytest.approx(200.0)
    assert math.isnan(per["fiber"])


def test_portion_scaling(table):
    result = table.portion("apple_pie", grams=150.0)
    assert result.nutrients["calories"] == pytest.approx(355.5)
    assert result.nutrients["protein"] == pytest.approx(3.6)
    assert result.label == "apple_pie"


def test_portion_display_rounds_and_nulls(table):
    display = table.portion("partial", grams=100.0).as_display()
    assert display["nutrients"]["calories"] == pytest.approx(200.0)
    assert display["nutrients"]["fiber"] is None


def test_unknown_label_raises_keyerror(table):
    with pytest.raises(KeyError, match="not found"):
        table.per_100g("does_not_exist")


def test_unknown_strategy_raises(table):
    with pytest.raises(ValueError, match="Unknown strategy"):
        table.per_100g("apple_pie", strategy="magic")


def test_negative_grams_raises(table):
    with pytest.raises(ValueError):
        table.portion("apple_pie", grams=-5)


def test_none_grams_raises(table):
    with pytest.raises(ValueError):
        table.portion("apple_pie", grams=None)
