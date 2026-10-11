"""Stage 3 tests for portion arithmetic."""

import math

import pytest

from foodmacros.nutrition.portion import (
    per_100g_scale,
    round_nutrients,
    scale_to_grams,
)


def test_per_100g_scale_from_basis_weight():
    result = per_100g_scale({"calories": 200.0, "protein": 10.0}, basis_weight=50.0)
    assert result["calories"] == pytest.approx(400.0)
    assert result["protein"] == pytest.approx(20.0)


def test_per_100g_scale_rejects_nonpositive_basis():
    with pytest.raises(ValueError):
        per_100g_scale({"calories": 100.0}, basis_weight=0)


def test_per_100g_scale_preserves_missing_values():
    result = per_100g_scale({"protein": float("nan")}, basis_weight=100.0)
    assert math.isnan(result["protein"])


def test_scale_to_grams():
    result = scale_to_grams({"calories": 237.0, "protein": 2.4}, grams=150.0)
    assert result["calories"] == pytest.approx(355.5)
    assert result["protein"] == pytest.approx(3.6)


def test_scale_to_grams_rejects_negative():
    with pytest.raises(ValueError):
        scale_to_grams({"calories": 100.0}, grams=-1)


def test_round_nutrients_turns_missing_into_none():
    result = round_nutrients({"calories": 355.55, "protein": float("nan")}, ndigits=1)
    assert result["calories"] == pytest.approx(355.6)
    assert result["protein"] is None
