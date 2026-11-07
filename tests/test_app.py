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


def test_table_rows_handles_gradio_dataframe_payload():
    module = _load_app()

    class _DataframePayload:
        data = [[1, "data:image/png;base64,abc", "pizza", "pizza", "90%", 150, 90, True]]

    assert module._table_rows(_DataframePayload()) == [
        [1, "data:image/png;base64,abc", "pizza", "pizza", "90%", 150, 90, True]
    ]


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


def test_multi_detect_returns_annotated_state_and_rows(
    synthetic_nutrition_csv, image_factory, tmp_path
):
    module = _load_app()
    pipeline = _make_multi(synthetic_nutrition_csv)
    image = str(image_factory(tmp_path / "plate.png", size=32))

    annotated, state, rows, summary, per_dish, report, payload, notes = module.multi_detect(
        pipeline, image, top_k=3, max_dishes=5, crop_padding=0.08
    )

    assert annotated is not None
    assert state is not None and len(state.dishes) == 2
    assert len(rows) == 2
    # [Dish, Thumbnail, Detected, Predicted, Confidence, Predicted, Edit, Include]
    assert rows[0][0] == 1 and rows[1][0] == 2
    assert rows[0][1].startswith("data:image/png;base64,")
    assert rows[0][3] == "pizza"  # predicted label
    assert rows[0][5] is None  # regression unavailable -> no predicted grams
    assert rows[0][6] is None  # measured override starts blank
    assert rows[0][7] is True  # include by default
    assert "Plate Summary" in summary
    assert "Nutrition by Dish" in per_dish
    assert "Calories" in per_dish
    assert payload["available"] is True


def test_multi_detect_populates_grams_with_automatic_prediction(
    synthetic_nutrition_csv, image_factory, tmp_path
):
    module = _load_app()
    pipeline = _make_multi(
        synthetic_nutrition_csv, regression=_stub_regression(mean=150.0)
    )
    image = str(image_factory(tmp_path / "plate.png", size=32))

    _, state, rows, summary, per_dish, report, payload, _ = module.multi_detect(
        pipeline, image, top_k=3, max_dishes=5, crop_padding=0.08
    )

    assert rows[0][5] == pytest.approx(150.0)
    assert rows[1][5] == pytest.approx(150.0)
    assert rows[0][6] is None and rows[1][6] is None
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

    annotated, state, rows, summary, per_dish, report, payload, notes = module.multi_detect(
        None, None, top_k=3, max_dishes=5, crop_padding=0.08
    )
    assert annotated is None and state is None and rows == []
    assert "No image provided" in report
    assert "No image provided" in summary
    assert "capture a plate" in per_dish


def test_multi_compute_applies_table_weights_and_totals(
    synthetic_nutrition_csv, image_factory, tmp_path
):
    module = _load_app()
    pipeline = _make_multi(synthetic_nutrition_csv)
    image = str(image_factory(tmp_path / "plate.png", size=32))
    _, state, rows, _, _, _, _, _ = module.multi_detect(
        pipeline, image, top_k=3, max_dishes=5, crop_padding=0.08
    )

    # Simulate the user editing "Edit Weight (g)" to 150 and 50, both included.
    table = [row[:] for row in rows]
    table[0][6] = 150
    table[1][6] = 50
    summary, per_dish, report, payload, notes = module.multi_compute(pipeline, state, table)

    assert payload["totals_basis"] == "per_predicted_weight"
    assert payload["total_grams"] == 200
    assert "200 g" in summary
    assert "Plate totals" in report


def test_multi_compute_keeps_automatic_weights_when_table_unchanged(
    synthetic_nutrition_csv, image_factory, tmp_path
):
    module = _load_app()
    pipeline = _make_multi(
        synthetic_nutrition_csv, regression=_stub_regression(mean=150.0)
    )
    image = str(image_factory(tmp_path / "plate.png", size=32))
    _, state, rows, _, _, _, _, _ = module.multi_detect(
        pipeline, image, top_k=3, max_dishes=5, crop_padding=0.08
    )

    summary, per_dish, report, payload, notes = module.multi_compute(pipeline, state, rows)

    assert payload["totals_basis"] == "per_predicted_weight"
    assert payload["total_grams"] == pytest.approx(300.0)
    assert [d["weight"]["status"] for d in payload["dishes"]] == [
        "estimated",
        "estimated",
    ]
    assert all(
        d["weight"]["method"] == "nutrition5k-regression" for d in payload["dishes"]
    )
    assert "300 g" in summary
    assert "Plate totals" in report


def test_multi_compute_manual_weight_overrides_automatic(
    synthetic_nutrition_csv, image_factory, tmp_path
):
    module = _load_app()
    pipeline = _make_multi(
        synthetic_nutrition_csv, regression=_stub_regression(mean=150.0)
    )
    image = str(image_factory(tmp_path / "plate.png", size=32))
    _, state, rows, _, _, _, _, _ = module.multi_detect(
        pipeline, image, top_k=3, max_dishes=5, crop_padding=0.08
    )

    table = [row[:] for row in rows]
    table[0][6] = 90  # explicit measured value overrides the prediction
    summary, per_dish, report, payload, notes = module.multi_compute(pipeline, state, table)

    assert payload["dishes"][0]["weight"]["status"] == "measured"
    assert payload["dishes"][0]["weight"]["grams"] == pytest.approx(90.0)
    assert payload["dishes"][1]["weight"]["status"] == "estimated"
    assert payload["dishes"][1]["weight"]["grams"] == pytest.approx(150.0)
    assert payload["total_grams"] == pytest.approx(240.0)
    assert "240 g" in summary
    assert "90 g" in per_dish
    assert "measured" in per_dish
    assert "Plate totals" in report

    # Recomputing must not corrupt the detection state held by the UI.
    assert state.dishes[0].weight.status == "estimated"
    assert state.dishes[0].weight.grams == pytest.approx(150.0)


def test_multi_compute_excludes_automatic_dish_from_total(
    synthetic_nutrition_csv, image_factory, tmp_path
):
    module = _load_app()
    pipeline = _make_multi(
        synthetic_nutrition_csv, regression=_stub_regression(mean=150.0)
    )
    image = str(image_factory(tmp_path / "plate.png", size=32))
    _, state, rows, _, _, _, _, _ = module.multi_detect(
        pipeline, image, top_k=3, max_dishes=5, crop_padding=0.08
    )

    table = [row[:] for row in rows]
    table[0][7] = False  # exclude dish 1 but keep its automatic weight shown
    summary, per_dish, report, payload, notes = module.multi_compute(pipeline, state, table)

    assert [d["included"] for d in payload["dishes"]] == [False, True]
    assert payload["dishes"][0]["weight"]["grams"] == pytest.approx(150.0)
    assert payload["total_grams"] == pytest.approx(150.0)
    assert "1 included" in summary
    assert "Excluded" in per_dish


def test_multi_compute_reports_missing_weight_when_unavailable(
    synthetic_nutrition_csv, image_factory, tmp_path
):
    module = _load_app()
    pipeline = _make_multi(synthetic_nutrition_csv)  # no regression
    image = str(image_factory(tmp_path / "plate.png", size=32))
    _, state, rows, _, _, _, _, _ = module.multi_detect(
        pipeline, image, top_k=3, max_dishes=5, crop_padding=0.08
    )

    assert rows[0][5] is None and rows[0][6] is None
    assert rows[1][5] is None and rows[1][6] is None
    summary, per_dish, report, payload, notes = module.multi_compute(pipeline, state, rows)

    assert payload["totals_basis"] == "none"
    assert payload["total_grams"] is None
    assert "missing weight" in payload["totals_reason"]
    assert "Incomplete" in summary
    assert "Weight unavailable" in per_dish


def test_multi_compute_respects_unchecked_include(
    synthetic_nutrition_csv, image_factory, tmp_path
):
    module = _load_app()
    pipeline = _make_multi(synthetic_nutrition_csv)
    image = str(image_factory(tmp_path / "plate.png", size=32))
    _, state, rows, _, _, _, _, _ = module.multi_detect(
        pipeline, image, top_k=3, max_dishes=5, crop_padding=0.08
    )

    table = [row[:] for row in rows]
    table[0][7] = False  # exclude dish 1
    table[0][6] = 150
    table[1][6] = 50
    summary, per_dish, report, payload, notes = module.multi_compute(pipeline, state, table)

    assert [d["included"] for d in payload["dishes"]] == [False, True]
    assert payload["total_grams"] == 50
    assert "50 g" in summary


def test_multi_compute_without_state_prompts_to_detect():
    module = _load_app()
    summary, per_dish, report, payload, notes = module.multi_compute(None, None, [])
    assert "Analyze Plate" in report
    assert "Analyze Plate" in summary
    assert "Analyze Plate" in per_dish
    assert payload == {}


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
