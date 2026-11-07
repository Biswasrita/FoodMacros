"""Integration tests for the RGB Nutrition5k weight regressor.

These cover three things:
- the bundled trained artifacts load through the pipeline resolver;
- a single-dish analysis turns a regression prediction into an ``estimated``
  weight and uses it for nutrition, while a measured value still overrides it;
- the multi-dish pipeline predicts one weight per dish crop and lets a
  measured per-dish value override the prediction.
"""

import numpy as np
import pytest

from foodmacros.config import AppConfig
from foodmacros.detection.yolo_detector import Detection, DetectionResult
from foodmacros.multi_pipeline import MultiDishPipeline
from foodmacros.nutrition.lookup import NutritionTable
from foodmacros.pipeline import (
    DEFAULT_REGRESSOR_DIR,
    FoodAnalysisPipeline,
    _load_regression,
    _resolve_regressor_dir,
)

BUNDLED_MODEL = DEFAULT_REGRESSOR_DIR / "mass_regressor.keras"


class _FakeClassifier:
    """Mimics the subset of the Keras model API used by ``predict_array``."""

    def __init__(self, probs):
        self._probs = np.asarray(probs, dtype=float)

    def predict(self, x, verbose=0):
        return self._probs[None, :]


class _ZeroMassModel:
    """A stub regressor whose normalised output is 0 -> denormalises to ``mean``."""

    def predict(self, x, verbose=0):
        return np.zeros((1, 1), dtype=np.float32)


class _FakeDetector:
    def __init__(self, boxes):
        self._boxes = boxes

    def detect(self, array):
        height, width = array.shape[:2]
        detections = [
            Detection("food", 0.9, (x1, y1, x2, y2))
            for (x1, y1, x2, y2) in self._boxes(width, height)
        ]
        return DetectionResult(True, detections)


def _stub_regression(mean=200.0, std=50.0, img_size=32):
    return {
        "model": _ZeroMassModel(),
        "preprocess_fn": lambda x: x,
        "img_size": img_size,
        "stats": {"total_mass": (mean, std)},
    }


@pytest.fixture
def config():
    cfg = AppConfig()
    cfg.model.img_size = 32
    return cfg


def test_bundled_regressor_loads_through_resolver():
    pytest.importorskip("tensorflow")
    if not BUNDLED_MODEL.exists():
        pytest.skip("bundled mass regressor artifact is not present")

    resolved = _resolve_regressor_dir()
    assert resolved == DEFAULT_REGRESSOR_DIR

    loaded = _load_regression(resolved, AppConfig())
    assert loaded is not None
    assert loaded["img_size"] == 224
    mean, std = loaded["stats"]["total_mass"]
    assert mean == pytest.approx(255.229, abs=1e-3)
    assert std == pytest.approx(221.912, abs=1e-3)


def test_single_dish_regression_estimates_weight_and_scales_nutrition(
    config, synthetic_nutrition_csv
):
    pipeline = FoodAnalysisPipeline(
        config=config,
        classifier=_FakeClassifier((0.1, 0.2, 0.7)),
        class_names=["apple_pie", "bibimbap", "pizza"],
        nutrition_table=NutritionTable.from_csv(synthetic_nutrition_csv),
        regression=_stub_regression(mean=200.0),
    )
    image = np.zeros((32, 32, 3), dtype=np.uint8)

    result = pipeline.analyze(image)

    assert result.prediction.label == "pizza"
    assert result.weight.status == "estimated"
    assert result.weight.grams == pytest.approx(200.0)
    assert result.weight.method == "nutrition5k-regression"
    assert result.nutrition.basis == "per_predicted_weight"
    assert result.nutrition.grams == pytest.approx(200.0)
    assert result.nutrition.nutrients["calories"] == pytest.approx(266 * 2.0)


def test_single_dish_measured_weight_overrides_regression(config, synthetic_nutrition_csv):
    pipeline = FoodAnalysisPipeline(
        config=config,
        classifier=_FakeClassifier((0.1, 0.2, 0.7)),
        class_names=["apple_pie", "bibimbap", "pizza"],
        nutrition_table=NutritionTable.from_csv(synthetic_nutrition_csv),
        regression=_stub_regression(mean=200.0),
    )
    image = np.zeros((32, 32, 3), dtype=np.uint8)

    result = pipeline.analyze(image, measured_grams=123)

    assert result.weight.status == "measured"
    assert result.weight.grams == 123
    assert result.nutrition.grams == 123


def test_multi_dish_predicts_weight_per_dish_and_measured_overrides(
    config, synthetic_nutrition_csv
):
    detector = _FakeDetector(
        lambda w, h: [(0, 0, w // 2, h), (w // 2, 0, w, h)]
    )
    pipeline = MultiDishPipeline(
        config=config,
        classifier=_FakeClassifier((0.1, 0.2, 0.7)),
        class_names=["apple_pie", "bibimbap", "pizza"],
        detector=detector,
        nutrition_table=NutritionTable.from_csv(synthetic_nutrition_csv),
        regression=_stub_regression(mean=150.0),
    )
    image = np.zeros((64, 64, 3), dtype=np.uint8)

    result = pipeline.analyze(image, max_dishes=2)

    assert result.available
    assert len(result.dishes) == 2
    for dish in result.dishes:
        assert dish.predicted_label == "pizza"
        assert dish.weight.status == "estimated"
        assert dish.weight.grams == pytest.approx(150.0)
        assert dish.nutrition.basis == "per_predicted_weight"

    assert result.totals_basis == "per_predicted_weight"
    assert result.total_grams == pytest.approx(300.0)

    updated = pipeline.apply_weights(result, per_dish_grams=[90, None])
    assert updated.dishes[0].weight.status == "measured"
    assert updated.dishes[0].weight.grams == 90
    assert updated.dishes[1].weight.status == "estimated"
    assert updated.dishes[1].weight.grams == pytest.approx(150.0)
    assert updated.total_grams == pytest.approx(240.0)


def test_apply_weights_does_not_mutate_detection_result(
    config, synthetic_nutrition_csv
):
    detector = _FakeDetector(
        lambda w, h: [(0, 0, w // 2, h), (w // 2, 0, w, h)]
    )
    pipeline = MultiDishPipeline(
        config=config,
        classifier=_FakeClassifier((0.1, 0.2, 0.7)),
        class_names=["apple_pie", "bibimbap", "pizza"],
        detector=detector,
        nutrition_table=NutritionTable.from_csv(synthetic_nutrition_csv),
        regression=_stub_regression(mean=150.0),
    )
    image = np.zeros((64, 64, 3), dtype=np.uint8)

    base = pipeline.analyze(image, max_dishes=2)
    updated = pipeline.apply_weights(base, per_dish_grams=[90, None])

    # The applied result reflects the override + automatic fallback...
    assert updated.dishes[0].weight.status == "measured"
    assert updated.dishes[0].weight.grams == pytest.approx(90.0)
    assert updated.dishes[1].weight.status == "estimated"
    assert updated.dishes[1].weight.grams == pytest.approx(150.0)
    # ...while the detection result kept in UI state is untouched.
    assert base.dishes[0].weight.status == "estimated"
    assert base.dishes[0].weight.grams == pytest.approx(150.0)
    assert base.dishes[1].weight.status == "estimated"
    assert base.dishes[1].weight.grams == pytest.approx(150.0)


def test_apply_weights_blank_reverts_to_automatic_regression(
    config, synthetic_nutrition_csv
):
    detector = _FakeDetector(
        lambda w, h: [(0, 0, w // 2, h), (w // 2, 0, w, h)]
    )
    pipeline = MultiDishPipeline(
        config=config,
        classifier=_FakeClassifier((0.1, 0.2, 0.7)),
        class_names=["apple_pie", "bibimbap", "pizza"],
        detector=detector,
        nutrition_table=NutritionTable.from_csv(synthetic_nutrition_csv),
        regression=_stub_regression(mean=150.0),
    )
    image = np.zeros((64, 64, 3), dtype=np.uint8)

    base = pipeline.analyze(image, max_dishes=2)
    manual = pipeline.apply_weights(base, per_dish_grams=[90, 80])
    assert all(d.weight.status == "measured" for d in manual.dishes)

    reverted = pipeline.apply_weights(manual, per_dish_grams=[None, None])

    assert all(d.weight.status == "estimated" for d in reverted.dishes)
    assert [d.weight.grams for d in reverted.dishes] == [
        pytest.approx(150.0),
        pytest.approx(150.0),
    ]
    assert reverted.total_grams == pytest.approx(300.0)


def test_multi_dish_defaults_load_bundled_regressor(config):
    pytest.importorskip("tensorflow")
    if not BUNDLED_MODEL.exists():
        pytest.skip("bundled mass regressor artifact is not present")

    pipeline = MultiDishPipeline.from_defaults(config)

    assert pipeline.regression is not None
    assert not any("No mass regressor" in note for note in pipeline.notes)
