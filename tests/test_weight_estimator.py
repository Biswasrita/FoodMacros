"""Stage 5 tests for the weight estimator status logic (no TF required)."""

import math
from dataclasses import dataclass

import pytest

from foodmacros.config import AppConfig
from foodmacros.weight.estimator import WeightEstimator


@dataclass
class FakeRegression:
    available: bool = True
    total_mass_g: float = 250.0
    calories: float = 400.0
    fat_g: float = 10.0
    carb_g: float = 50.0
    protein_g: float = 20.0


@pytest.fixture
def estimator():
    return WeightEstimator(AppConfig())


def test_measured_weight_is_accepted(estimator):
    result = estimator.estimate(measured_grams=250.0)
    assert result.status == "measured"
    assert result.available
    assert result.grams == pytest.approx(250.0)
    assert result.confidence == "high"


def test_measured_weight_invalid_values(estimator):
    assert estimator.estimate(measured_grams=-5).status == "unavailable"
    assert estimator.estimate(measured_grams="abc").status == "unavailable"


def test_measured_weight_outside_range_warns(estimator):
    result = estimator.estimate(measured_grams=5000.0)
    assert result.status == "measured"
    assert result.warnings


def test_no_sources_is_unavailable(estimator):
    result = estimator.estimate()
    assert result.status == "unavailable"
    assert result.grams is None
    assert not result.available
    assert result.warnings


def test_regression_reliable_is_estimated(estimator):
    result = estimator.estimate(regression=FakeRegression(total_mass_g=250.0))
    assert result.status == "estimated"
    assert result.grams == pytest.approx(250.0)
    assert result.details["calories"] == pytest.approx(400.0)


def test_regression_out_of_range_is_unavailable(estimator):
    result = estimator.estimate(regression=FakeRegression(total_mass_g=5000.0))
    assert result.status == "unavailable"


def test_regression_non_finite_is_unavailable(estimator):
    result = estimator.estimate(regression=FakeRegression(total_mass_g=float("nan")))
    assert result.status == "unavailable"


def test_depth_disabled_by_default(estimator):
    evidence = estimator.depth_evidence_from(200.0, "pizza")
    assert evidence.available
    result = estimator.estimate(depth=evidence)
    assert result.status == "unavailable"
    assert any("disabled" in w for w in result.warnings)


def test_depth_experimental_when_enabled():
    config = AppConfig()
    config.weight.allow_experimental_depth = True
    estimator = WeightEstimator(config)
    evidence = estimator.depth_evidence_from(200.0, "pizza")
    result = estimator.estimate(depth=evidence)
    assert result.status == "experimental"
    assert result.grams == pytest.approx(120.0)  # 200 cm^3 * 0.6 g/cm^3
    assert result.confidence == "low"


def test_depth_unknown_density_is_unavailable():
    config = AppConfig()
    config.weight.allow_experimental_depth = True
    estimator = WeightEstimator(config)
    evidence = estimator.depth_evidence_from(200.0, "unobtainium")
    assert not evidence.available
    result = estimator.estimate(depth=evidence)
    assert result.status == "unavailable"


def test_depth_evidence_rejects_nonpositive_volume(estimator):
    evidence = estimator.depth_evidence_from(0.0, "pizza")
    assert not evidence.available


def test_measured_takes_precedence_over_regression(estimator):
    result = estimator.estimate(measured_grams=300.0, regression=FakeRegression(total_mass_g=100.0))
    assert result.status == "measured"
    assert result.grams == pytest.approx(300.0)
