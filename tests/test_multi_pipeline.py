"""Tests for the multi-dish pipeline, reporting, and nutrition aliases."""

import json

import numpy as np
import pytest

from foodmacros.config import AppConfig
from foodmacros.detection.yolo_detector import Detection, DetectionResult
from foodmacros.multi_pipeline import MultiDishPipeline
from foodmacros.multi_reporting import format_multi_analysis, multi_analysis_to_dict
from foodmacros.nutrition.lookup import NutritionTable


class _IndexModel:
    """Keras-like model returning a fixed argmax class."""

    def __init__(self, n_classes, index=2):
        self.n_classes = n_classes
        self.index = index

    def predict(self, x, verbose=0):
        probs = np.full((x.shape[0], self.n_classes), 0.01, dtype=float)
        probs[:, self.index] = 0.9
        return probs


class _FakeDetector:
    def __init__(self, detections):
        self._detections = list(detections)

    def detect(self, image):
        return DetectionResult(True, detections=list(self._detections))


class _UnavailableDetector:
    def detect(self, image):
        return DetectionResult(False, reason="no weights")


CLASSES = ["apple_pie", "bibimbap", "pizza"]


@pytest.fixture
def config():
    cfg = AppConfig()
    cfg.model.img_size = 32
    return cfg


def _make(config, nutrition_csv, detector):
    return MultiDishPipeline(
        config=config,
        classifier=_IndexModel(len(CLASSES), index=2),
        class_names=CLASSES,
        detector=detector,
        nutrition_table=NutritionTable.from_csv(nutrition_csv),
    )


def _two_boxes():
    return [
        Detection("pizza", 0.9, (2, 2, 14, 14)),
        Detection("salad fresh", 0.8, (16, 16, 30, 30)),
    ]


def test_two_dishes_without_weights_have_no_total(
    config, synthetic_nutrition_csv, image_factory, tmp_path
):
    image = image_factory(tmp_path / "plate.png", size=32)
    pipeline = _make(config, synthetic_nutrition_csv, _FakeDetector(_two_boxes()))

    result = pipeline.analyze(str(image))

    assert result.available
    assert len(result.dishes) == 2
    assert all(d.predicted_label == "pizza" for d in result.dishes)
    # No weights -> no plate total.
    assert result.totals_basis == "none"
    assert result.totals == {}
    assert "missing weight" in result.totals_reason
    # ... but each dish still shows reference-quantity macros.
    assert result.dishes[0].nutrition.basis == "per_reference"
    assert result.dishes[0].nutrition.nutrients["calories"] == pytest.approx(266)


def test_per_dish_weights_produce_a_total(config, synthetic_nutrition_csv, image_factory, tmp_path):
    image = image_factory(tmp_path / "plate.png", size=32)
    pipeline = _make(config, synthetic_nutrition_csv, _FakeDetector(_two_boxes()))

    result = pipeline.analyze(str(image), per_dish_grams=[150, 50])

    assert all(d.weight.status == "measured" for d in result.dishes)
    assert result.totals_basis == "per_predicted_weight"
    assert result.total_grams == 200
    # pizza at 150 g + pizza at 50 g = 266 * 2.0
    assert result.totals["calories"] == pytest.approx(266 * 2.0)


def test_partial_weights_give_no_total_and_a_reason(
    config, synthetic_nutrition_csv, image_factory, tmp_path
):
    image = image_factory(tmp_path / "plate.png", size=32)
    pipeline = _make(config, synthetic_nutrition_csv, _FakeDetector(_two_boxes()))

    result = pipeline.analyze(str(image), per_dish_grams=[150, None])

    assert result.totals_basis == "none"
    assert result.totals == {}
    assert "dish 2" in result.totals_reason
    assert result.dishes[0].nutrition.basis == "per_predicted_weight"
    assert result.dishes[1].nutrition.basis == "per_reference"


def test_apply_weights_two_phase(config, synthetic_nutrition_csv, image_factory, tmp_path):
    image = image_factory(tmp_path / "plate.png", size=32)
    pipeline = _make(config, synthetic_nutrition_csv, _FakeDetector(_two_boxes()))

    base = pipeline.analyze(str(image))  # detect + classify, no weights
    assert base.totals_basis == "none"

    updated = pipeline.apply_weights(base, [100, 100])
    assert updated.totals_basis == "per_predicted_weight"
    assert updated.total_grams == 200
    assert updated.totals["calories"] == pytest.approx(266 * 2.0)
    # The original result is not mutated.
    assert base.totals_basis == "none"


def test_include_flag_excludes_a_dish_from_the_total(
    config, synthetic_nutrition_csv, image_factory, tmp_path
):
    image = image_factory(tmp_path / "plate.png", size=32)
    pipeline = _make(config, synthetic_nutrition_csv, _FakeDetector(_two_boxes()))

    result = pipeline.analyze(
        str(image), per_dish_grams=[100, 100], include=[True, False]
    )

    assert result.included == [True, False]
    assert result.totals_basis == "per_predicted_weight"
    # Only dish 1 (100 g) counts.
    assert result.totals["calories"] == pytest.approx(266)
    assert result.total_grams == 100


def test_detector_unavailable_falls_back_to_whole_image(
    config, synthetic_nutrition_csv, image_factory, tmp_path
):
    image = image_factory(tmp_path / "plate.png", size=32)
    pipeline = _make(config, synthetic_nutrition_csv, _UnavailableDetector())

    result = pipeline.analyze(str(image))

    assert result.available
    assert len(result.dishes) == 1
    assert result.dishes[0].detection is None
    assert result.dishes[0].predicted_label == "pizza"
    assert any("Detection unavailable" in n for n in result.notes)


def test_measured_weight_used_for_single_dish(
    config, synthetic_nutrition_csv, image_factory, tmp_path
):
    image = image_factory(tmp_path / "plate.png", size=32)
    pipeline = _make(config, synthetic_nutrition_csv, _UnavailableDetector())

    result = pipeline.analyze(str(image), measured_grams=150)

    dish = result.dishes[0]
    assert dish.weight.status == "measured"
    assert dish.nutrition.basis == "per_predicted_weight"
    assert dish.nutrition.nutrients["calories"] == pytest.approx(266 * 1.5)
    assert result.total_grams == 150
    assert result.totals_basis == "per_predicted_weight"


def test_measured_weight_ignored_for_multiple_dishes(
    config, synthetic_nutrition_csv, image_factory, tmp_path
):
    image = image_factory(tmp_path / "plate.png", size=32)
    pipeline = _make(config, synthetic_nutrition_csv, _FakeDetector(_two_boxes()))

    result = pipeline.analyze(str(image), measured_grams=300)

    assert any("cannot be split" in w for w in result.warnings)
    assert all(d.weight.status == "unavailable" for d in result.dishes)


def test_max_dishes_caps_analysis(config, synthetic_nutrition_csv, image_factory, tmp_path):
    image = image_factory(tmp_path / "plate.png", size=32)
    pipeline = _make(config, synthetic_nutrition_csv, _FakeDetector(_two_boxes()))

    result = pipeline.analyze(str(image), max_dishes=1)

    assert len(result.dishes) == 1
    assert any("max_dishes" in w for w in result.warnings)


def test_from_defaults_reports_missing_artifacts(config, tmp_path):
    pipeline = MultiDishPipeline.from_defaults(
        config,
        model_dir=tmp_path / "empty",
        detector_weights=tmp_path / "nope.pt",
        nutrition_csv=tmp_path / "nope.csv",
    )
    assert any("classifier" in n.lower() for n in pipeline.notes)
    assert any("nutrition" in n.lower() for n in pipeline.notes)


def test_result_is_json_serialisable_and_formatted(
    config, synthetic_nutrition_csv, image_factory, tmp_path
):
    image = image_factory(tmp_path / "plate.png", size=32)
    pipeline = _make(config, synthetic_nutrition_csv, _FakeDetector(_two_boxes()))

    result = pipeline.analyze(str(image))
    payload = multi_analysis_to_dict(result)
    json.dumps(payload)  # must not raise
    assert payload["available"] is True
    assert len(payload["dishes"]) == 2

    text = format_multi_analysis(result)
    assert "=== Multi-Dish Food Analysis ===" in text
    assert "Plate totals" in text
    assert "Limitations" in text


def test_nutrition_aliases_resolve_display_names(tmp_path):
    csv_path = tmp_path / "alias.csv"
    header = "label,weight,calories,protein,carbohydrates,fats,fiber,sugars,sodium"
    rows = [
        "Cupcakes,100,305,4,36,16,1,22,235",
        "Fish and Chips,100,280,15,30,12,3,1,500",
    ]
    csv_path.write_text("\n".join([header] + rows), encoding="utf-8")
    table = NutritionTable.from_csv(csv_path)

    assert table.has_label("cup_cakes")
    assert table.has_label("fish_and_chips")
    assert table.per_100g("cup_cakes")["calories"] == pytest.approx(305)
