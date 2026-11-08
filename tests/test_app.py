"""Stage 7 tests for the Gradio app module.

The app must be importable *without* gradio installed; only ``build_interface``
and ``main`` require it. We therefore test import surface and the pure helpers.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app" / "gradio_app.py"


def _load_app():
    spec = importlib.util.spec_from_file_location("foodmacros_gradio_app", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_app_module_imports_without_gradio():
    module = _load_app()
    assert callable(module.build_interface)
    assert callable(module.main)
    assert callable(module.create_pipeline)
    assert str(ROOT / "src") in sys.path


def test_app_helpers_handle_unavailable_result():
    module = _load_app()
    from foodmacros.pipeline import AnalysisResult, NutritionOutcome
    from foodmacros.weight.estimator import WeightEstimate

    result = AnalysisResult(
        prediction=None,
        detection=None,
        weight=WeightEstimate("unavailable", None, "none", None, ["nothing available"]),
        nutrition=NutritionOutcome(False, reason="no data"),
        reference_quantity_g=100.0,
    )
    assert module._classification_text(result).startswith("Unavailable")
    assert module._macros_text(result).startswith("Unavailable")
    assert "Unavailable" in module._single_prediction_html(result)
    assert "Unavailable" in module._single_weight_html(result)
    assert "Unavailable" in module._single_nutrition_html(result)


def test_app_main_reports_missing_gradio():
    module = _load_app()
    from foodmacros.env_checks import has_package

    if has_package("gradio"):
        return  # environment has gradio; the guard path is not exercised
    assert module.main(["--config", str(ROOT / "configs" / "default.json")]) == 1


class _FixedModel:
    def __init__(self, n_classes, index):
        self.n_classes = n_classes
        self.index = index

    def predict(self, x, verbose=0):
        import numpy as np

        probs = np.full((x.shape[0], self.n_classes), 0.01, dtype=float)
        probs[:, self.index] = 0.9
        return probs


class _FixedDetector:
    def __init__(self, boxes):
        self._boxes = list(boxes)

    def detect(self, image):
        from foodmacros.detection.yolo_detector import DetectionResult

        return DetectionResult(True, detections=list(self._boxes))


class _ZeroMassModel:
    """Regressor stub whose normalised output denormalises to ``mean`` grams."""

    def predict(self, x, verbose=0):
        import numpy as np

        return np.zeros((1, 1), dtype=np.float32)


def _stub_regression(mean=150.0, std=50.0, img_size=32):
    return {
        "model": _ZeroMassModel(),
        "preprocess_fn": lambda x: x,
        "img_size": img_size,
        "stats": {"total_mass": (mean, std)},
    }


def _make_multi(synthetic_nutrition_csv, regression=None):
    from foodmacros.config import AppConfig
    from foodmacros.detection.yolo_detector import Detection
    from foodmacros.multi_pipeline import MultiDishPipeline
    from foodmacros.nutrition.lookup import NutritionTable

    cfg = AppConfig()
    cfg.model.img_size = 32
    classes = ["apple_pie", "bibimbap", "pizza"]
    boxes = [Detection("pizza", 0.9, (2, 2, 14, 14)), Detection("salad fresh", 0.8, (16, 16, 30, 30))]
    return MultiDishPipeline(
        config=cfg,
        classifier=_FixedModel(len(classes), index=2),
        class_names=classes,
        detector=_FixedDetector(boxes),
        nutrition_table=NutritionTable.from_csv(synthetic_nutrition_csv),
        regression=regression,
    )


def test_multi_detect_returns_annotated_summary_and_cards(
    synthetic_nutrition_csv, image_factory, tmp_path
):
    module = _load_app()
    pipeline = _make_multi(synthetic_nutrition_csv)
    image = str(image_factory(tmp_path / "plate.png", size=32))

    annotated, summary, per_dish, report, payload, notes = module.multi_detect(
        pipeline, image, top_k=3, max_dishes=5, crop_padding=0.08
    )

    assert annotated is not None
    assert "Plate Summary" in summary
    assert "Nutrition by Dish" in per_dish
    assert "Calories" in per_dish
    assert "Weight unavailable" in per_dish
    assert payload["available"] is True
    assert "Plate totals" in report


def test_multi_detect_populates_automatic_weights_in_cards(
    synthetic_nutrition_csv, image_factory, tmp_path
):
    module = _load_app()
    pipeline = _make_multi(
        synthetic_nutrition_csv, regression=_stub_regression(mean=150.0)
    )
    image = str(image_factory(tmp_path / "plate.png", size=32))

    _, summary, per_dish, report, payload, _ = module.multi_detect(
        pipeline, image, top_k=3, max_dishes=5, crop_padding=0.08
    )

    assert [d["weight"]["status"] for d in payload["dishes"]] == [
        "estimated",
        "estimated",
    ]
    assert payload["total_grams"] == pytest.approx(300.0)
    assert "300 g" in summary
    assert "150 g" in per_dish
    assert "estimated" in per_dish
    assert "Plate totals" in report


def test_multi_detect_without_image_reports_error():
    module = _load_app()

    annotated, summary, per_dish, report, payload, notes = module.multi_detect(
        None, None, top_k=3, max_dishes=5, crop_padding=0.08
    )
    assert annotated is None and payload == {}
    assert "No image provided" in report
    assert "No image provided" in summary
    assert "capture a plate" in per_dish


def test_multi_ui_has_no_table_or_recompute_controls():
    module = _load_app()

    assert not hasattr(module, "multi_compute")
    assert not hasattr(module, "_table_rows")
    assert "fm-table-card" not in module.CSS
    assert "fm-tip-row" not in module.CSS


def test_styles_cover_about_text_and_dish_cards():
    module = _load_app()

    assert ".fm-about-copy .prose" in module.CSS
    assert ".fm-dish-card" in module.CSS
    assert "--fm-muted: #475569" in module.CSS


def test_build_interface_smoke(synthetic_nutrition_csv):
    pytest.importorskip("gradio")
    module = _load_app()

    class _SingleStub:
        def __init__(self):
            from foodmacros.config import AppConfig

            self.config = AppConfig()

    demo = module.build_interface(
        _SingleStub(), multi_pipeline=_make_multi(synthetic_nutrition_csv)
    )
    assert demo is not None
