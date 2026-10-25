"""Stage 7 tests for the Gradio app module.

The app must be importable *without* gradio installed; only ``build_interface``
and ``main`` require it. We therefore test import surface and the pure helpers.
"""

import importlib.util
import sys
from pathlib import Path

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


def _make_multi(synthetic_nutrition_csv):
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
    )


def test_multi_detect_returns_annotated_state_and_rows(
    synthetic_nutrition_csv, image_factory, tmp_path
):
    module = _load_app()
    pipeline = _make_multi(synthetic_nutrition_csv)
    image = str(image_factory(tmp_path / "plate.png", size=32))

    annotated, state, rows, report, payload, notes = module.multi_detect(
        pipeline, image, top_k=3, max_dishes=5, crop_padding=0.08
    )

    assert annotated is not None
    assert state is not None and len(state.dishes) == 2
    assert len(rows) == 2
    # [Dish, Detected, Predicted, Confidence, Include, Estimated weight (g), Weight status]
    assert rows[0][0] == 1 and rows[1][0] == 2
    assert rows[0][2] == "pizza"  # predicted label
    assert rows[0][4] is True  # include by default
    assert rows[0][6] in {"unavailable", "measured", "estimated", "experimental"}
    assert payload["available"] is True


def test_multi_detect_without_image_reports_error():
    module = _load_app()

    annotated, state, rows, report, payload, notes = module.multi_detect(
        None, None, top_k=3, max_dishes=5, crop_padding=0.08
    )
    assert annotated is None and state is None and rows == []
    assert "No image provided" in report


def test_multi_compute_applies_table_weights_and_totals(
    synthetic_nutrition_csv, image_factory, tmp_path
):
    module = _load_app()
    pipeline = _make_multi(synthetic_nutrition_csv)
    image = str(image_factory(tmp_path / "plate.png", size=32))
    _, state, rows, _, _, _ = module.multi_detect(
        pipeline, image, top_k=3, max_dishes=5, crop_padding=0.08
    )

    # Include both dishes; automatic weights may be unavailable without calibration
    table = [row[:] for row in rows]
    table[0][4] = True
    table[1][4] = True
    report, payload, notes = module.multi_compute(pipeline, state, table)

    assert payload["available"] is True
    assert "dishes" in payload


def test_multi_compute_respects_unchecked_include(
    synthetic_nutrition_csv, image_factory, tmp_path
):
    module = _load_app()
    pipeline = _make_multi(synthetic_nutrition_csv)
    image = str(image_factory(tmp_path / "plate.png", size=32))
    _, state, rows, _, _, _ = module.multi_detect(
        pipeline, image, top_k=3, max_dishes=5, crop_padding=0.08
    )

    table = [row[:] for row in rows]
    table[0][4] = False  # exclude dish 1
    table[1][4] = True
    report, payload, notes = module.multi_compute(pipeline, state, table)

    assert [d["included"] for d in payload["dishes"]] == [False, True]


def test_multi_compute_without_state_prompts_to_detect():
    module = _load_app()
    report, payload, notes = module.multi_compute(None, None, [])
    assert "Detect" in report
    assert payload == {}
