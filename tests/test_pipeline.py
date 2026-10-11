"""Stage 7 tests for the end-to-end pipeline and result reporting."""

import json

import numpy as np
import pytest

from foodmacros.config import AppConfig
from foodmacros.nutrition.lookup import NutritionTable
from foodmacros.pipeline import AnalysisResult, FoodAnalysisPipeline, NutritionOutcome
from foodmacros.reporting import analysis_to_dict, format_analysis, format_nutrients


class _FakeModel:
    """Mimics the subset of the Keras model API used by ``predict_array``."""

    def __init__(self, probs):
        self._probs = np.asarray(probs, dtype=float)

    def predict(self, x, verbose=0):
        return self._probs[None, :]


def _make_pipeline(config, nutrition_csv, probs=(0.1, 0.2, 0.7)):
    return FoodAnalysisPipeline(
        config=config,
        classifier=_FakeModel(probs),
        class_names=["apple_pie", "bibimbap", "pizza"],
        nutrition_table=NutritionTable.from_csv(nutrition_csv),
    )


@pytest.fixture
def config():
    cfg = AppConfig()
    cfg.model.img_size = 32
    return cfg


def test_measured_weight_drives_macros(config, synthetic_nutrition_csv, image_factory, tmp_path):
    image = image_factory(tmp_path / "meal.png")
    pipeline = _make_pipeline(config, synthetic_nutrition_csv)

    result = pipeline.analyze(str(image), measured_grams=150)

    assert result.prediction.available
    assert result.prediction.label == "pizza"
    assert result.weight.status == "measured"
    assert result.weight.grams == 150
    assert result.nutrition.basis == "per_predicted_weight"
    assert result.nutrition.grams == 150
    assert result.nutrition.nutrients["calories"] == pytest.approx(266 * 1.5)


def test_no_weight_source_reports_reference_quantity(config, synthetic_nutrition_csv, image_factory, tmp_path):
    image = image_factory(tmp_path / "meal.png")
    pipeline = _make_pipeline(config, synthetic_nutrition_csv)

    result = pipeline.analyze(str(image))

    assert not result.weight.available
    assert result.weight.status == "unavailable"
    assert result.nutrition.available
    assert result.nutrition.basis == "per_reference"
    assert result.nutrition.grams == config.nutrition.reference_quantity_g
    assert result.nutrition.nutrients["calories"] == pytest.approx(266)


def test_missing_classifier_degrades_cleanly(config, synthetic_nutrition_csv):
    pipeline = FoodAnalysisPipeline(
        config=config, nutrition_table=NutritionTable.from_csv(synthetic_nutrition_csv)
    )
    result = pipeline.analyze(None)

    assert not result.prediction.available
    assert result.prediction.reason == "classifier not loaded"
    assert not result.nutrition.available
    assert "no predicted label" in result.nutrition.reason
    assert any("classifier" in n.lower() for n in result.notes)


def test_detection_unavailable_without_detector(config, synthetic_nutrition_csv, image_factory, tmp_path):
    image = image_factory(tmp_path / "meal.png")
    pipeline = _make_pipeline(config, synthetic_nutrition_csv)

    result = pipeline.analyze(str(image))

    assert result.detection is not None
    assert not result.detection.available
    assert "no detector configured" in result.detection.reason


def test_unknown_label_yields_explicit_reason(config, synthetic_nutrition_csv, image_factory, tmp_path):
    image = image_factory(tmp_path / "meal.png")
    pipeline = FoodAnalysisPipeline(
        config=config,
        classifier=_FakeModel((0.9, 0.05, 0.05)),
        class_names=["apple_pie", "bibimbap", "pizza"],
        nutrition_table=NutritionTable.from_csv(synthetic_nutrition_csv),
    )
    pipeline.class_names = ["sushi", "bibimbap", "pizza"]

    result = pipeline.analyze(str(image), measured_grams=100)

    assert result.prediction.label == "sushi"
    assert not result.nutrition.available
    assert "not in nutrition table" in result.nutrition.reason


def test_result_is_json_serialisable(config, synthetic_nutrition_csv, image_factory, tmp_path):
    image = image_factory(tmp_path / "meal.png")
    result = _make_pipeline(config, synthetic_nutrition_csv).analyze(str(image), measured_grams=120)
    payload = analysis_to_dict(result)
    json.dumps(payload)  # must not raise
    assert payload["classification"]["label"] == "pizza"
    assert payload["weight"]["grams"] == 120


def test_format_analysis_mentions_all_sections(config, synthetic_nutrition_csv, image_factory, tmp_path):
    image = image_factory(tmp_path / "meal.png")
    result = _make_pipeline(config, synthetic_nutrition_csv).analyze(str(image), measured_grams=150)
    text = format_analysis(result)

    assert "=== Food Analysis ===" in text
    assert "Classification : pizza" in text
    assert "Detection" in text
    assert "Weight" in text
    assert "Calories" in text
    assert "Limitations" in text


# -- default runtime artifacts --------------------------------------------

def test_default_artifact_paths_exist():
    from foodmacros.pipeline import DEFAULT_CLASSIFIER_DIR, DEFAULT_DETECTOR_WEIGHTS
    from foodmacros.multi_pipeline import DEFAULT_NUTRITION_CSV

    assert (DEFAULT_CLASSIFIER_DIR / "classifier.keras").exists()
    assert (DEFAULT_CLASSIFIER_DIR / "class_names.json").exists()
    assert DEFAULT_DETECTOR_WEIGHTS.exists()
    assert DEFAULT_NUTRITION_CSV.exists()


def test_resolver_uses_bundled_defaults_when_nothing_discovered(config):
    from pathlib import Path

    from foodmacros.data.discovery import ResolvedInputs
    from foodmacros.pipeline import (
        DEFAULT_CLASSIFIER_DIR,
        DEFAULT_DETECTOR_WEIGHTS,
        _resolve_default_artifacts,
    )

    resolved = ResolvedInputs(
        notes=[
            "No YOLO food-detection weights (*.pt) found: detection will report "
            "'unavailable' until compatible weights are provided."
        ]
    )
    model_dir, detection_weights, notes = _resolve_default_artifacts(config, resolved)

    assert model_dir == DEFAULT_CLASSIFIER_DIR
    assert detection_weights == str(DEFAULT_DETECTOR_WEIGHTS)
    assert all("detection weight" not in n.lower() for n in notes)


def test_resolver_keeps_explicit_paths(config, tmp_path):
    from foodmacros.data.discovery import ResolvedInputs
    from foodmacros.pipeline import _resolve_default_artifacts

    explicit_model = tmp_path / "my_model"
    explicit_det = tmp_path / "my_detector.pt"
    explicit_model.mkdir()
    (explicit_model / "classifier.keras").write_bytes(b"x")
    (explicit_model / "class_names.json").write_text('["a"]', encoding="utf-8")
    explicit_det.write_bytes(b"x")

    resolved = ResolvedInputs(detection_weights=str(explicit_det), notes=[])
    model_dir, detection_weights, notes = _resolve_default_artifacts(
        config, resolved, model_dir=str(explicit_model)
    )

    assert model_dir == explicit_model
    assert detection_weights == str(explicit_det)


def test_resolver_skips_detector_when_disabled(config):
    from foodmacros.data.discovery import ResolvedInputs
    from foodmacros.pipeline import _resolve_default_artifacts

    config.detection.enabled = False
    model_dir, detection_weights, notes = _resolve_default_artifacts(
        config, ResolvedInputs(notes=[])
    )
    assert detection_weights is None


def test_bundled_detector_matches_only_the_bundle(config):
    from foodmacros.pipeline import (
        DEFAULT_DETECTOR_WEIGHTS,
        _is_bundled_detector,
    )

    assert _is_bundled_detector(str(DEFAULT_DETECTOR_WEIGHTS))
    assert not _is_bundled_detector(str(DEFAULT_DETECTOR_WEIGHTS) + ".bak")
    assert not _is_bundled_detector(None)
    assert not _is_bundled_detector("C:\\somewhere\\other.pt")


def test_format_nutrients_renders_known_and_extra_keys():
    text = format_nutrients({"calories": 100.0, "protein": None, "custom": 3.14159})
    assert "Calories (kcal): 100" in text
    assert "Protein (g): n/a" in text
    assert "custom: 3.1" in text


def test_nutrition_all_none_still_marked_unavailable(config):
    outcome = NutritionOutcome(False, reason="nothing")
    assert outcome.basis == "none"
    assert not outcome.available
