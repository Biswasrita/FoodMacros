"""Interactive Gradio application for the FoodMacros pipeline.

Gradio is imported lazily so the rest of the project works without it. Launch
locally by default; ``--share`` (or ``ui.share`` in config) enables a public
tunnel and failures are reported gracefully instead of crashing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from foodmacros.config import AppConfig, load_config  # noqa: E402
from foodmacros.detection.visualize import draw_detections  # noqa: E402
from foodmacros.env_checks import has_package  # noqa: E402
from foodmacros.multi_pipeline import MultiDishPipeline  # noqa: E402
from foodmacros.multi_reporting import (  # noqa: E402
    format_multi_analysis,
    multi_analysis_to_dict,
)
from foodmacros.pipeline import FoodAnalysisPipeline  # noqa: E402
from foodmacros.reporting import (  # noqa: E402
    LIMITATIONS,
    analysis_to_dict,
    format_nutrients,
)


def create_pipeline(config: AppConfig, model_dir=None, regressor_dir=None):
    pipeline, resolved = FoodAnalysisPipeline.from_discovery(
        config, model_dir=model_dir, mass_regressor_dir=regressor_dir
    )
    return pipeline, resolved


def create_multi_pipeline(config: AppConfig, model_dir=None):
    """Build the multi-dish pipeline from the project's default artifacts."""
    return MultiDishPipeline.from_defaults(config, model_dir=model_dir)


def _classification_text(result) -> str:
    prediction = result.prediction
    if prediction is None or not prediction.available:
        reason = prediction.reason if prediction else "not run"
        return f"Unavailable: {reason}"
    lines = [f"{prediction.label}  ({prediction.confidence:.1%})"]
    for name, prob in prediction.top_k:
        lines.append(f"  {name}: {prob:.1%}")
    return "\n".join(lines)


def _macros_text(result) -> str:
    if not result.nutrition.available:
        return f"Unavailable: {result.nutrition.reason}"
    header = f"{result.nutrition.label} — {result.nutrition.basis} ({result.nutrition.grams:g} g)"
    return header + "\n" + format_nutrients(result.nutrition.nutrients)


def _clean_measured(measured):
    if measured is None or str(measured).strip() == "":
        return None
    try:
        candidate = float(measured)
    except (TypeError, ValueError):
        return None
    return candidate if candidate > 0 else None


MULTI_HEADERS = ["Dish", "Detected", "Predicted", "Confidence", "Include", "Grams (g)"]
MULTI_DATATYPE = ["number", "str", "str", "str", "bool", "number"]


def _image_input(gr, label):
    """An image input with upload + camera, tolerant of older Gradio APIs."""
    try:
        return gr.Image(type="filepath", sources=["upload", "webcam"], label=label)
    except TypeError:
        return gr.Image(type="filepath", label=label)


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "y", "x", "include")
    return bool(value)


def _as_grams(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        grams = float(text)
    except (TypeError, ValueError):
        return None
    return grams if grams > 0 else None


def _table_rows(table):
    if table is None:
        return []
    if hasattr(table, "values"):  # pandas DataFrame
        return [list(row) for row in table.values.tolist()]
    if hasattr(table, "tolist"):
        return table.tolist()
    return list(table)


def multi_detect(multi_pipeline, image, top_k, max_dishes, crop_padding):
    """Detect + classify a plate and return (annotated image, state, rows, ...).

    ``rows`` is the dish table the user sees: ``[Dish, Detected, Predicted,
    Confidence, Include, Grams]``. Gradio-ready: the state carries the full
    result so weights can be applied later without re-detecting.
    """
    if image is None:
        return None, None, [], "No image provided.", {}, LIMITATIONS
    result = multi_pipeline.analyze(
        image,
        top_k=int(top_k),
        max_dishes=int(max_dishes),
        crop_padding=float(crop_padding),
    )
    if not result.available:
        return None, None, [], f"Analysis unavailable: {result.reason}", {}, LIMITATIONS

    detections = [d.detection for d in result.dishes if d.detection is not None]
    try:
        annotated = draw_detections(image, detections) if detections else image
    except Exception:
        annotated = image

    rows = []
    for dish in result.dishes:
        det = dish.detection
        pred = dish.prediction
        rows.append(
            [
                dish.index,
                det.label if det else "whole image",
                pred.label if (pred is not None and pred.available) else "",
                (
                    f"{pred.confidence:.1%}"
                    if (pred is not None and pred.available and pred.confidence is not None)
                    else ""
                ),
                True,
                None,
            ]
        )
    return (
        annotated,
        result,
        rows,
        format_multi_analysis(result),
        multi_analysis_to_dict(result),
        LIMITATIONS,
    )


def multi_compute(multi_pipeline, state, table):
    """Read the dish table's Include/Grams columns and recompute nutrition."""
    if state is None:
        return (
            "No detection result yet. Click '1) Detect & classify' first.",
            {},
            LIMITATIONS,
        )
    rows = _table_rows(table)
    grams_by_index = {}
    include_by_index = {}
    for row in rows:
        if not row:
            continue
        try:
            idx = int(float(row[0]))
        except (TypeError, ValueError, IndexError):
            continue
        include_by_index[idx] = _as_bool(row[4]) if len(row) > 4 else True
        grams_by_index[idx] = _as_grams(row[5]) if len(row) > 5 else None

    per_dish_grams = [grams_by_index.get(d.index) for d in state.dishes]
    include = [include_by_index.get(d.index, True) for d in state.dishes]
    updated = multi_pipeline.apply_weights(state, per_dish_grams, include)
    return format_multi_analysis(updated), multi_analysis_to_dict(updated), LIMITATIONS


def build_interface(pipeline: FoodAnalysisPipeline, multi_pipeline=None):
    import gradio as gr

    def run(image, measured, top_k):
        if image is None:
            return "No image provided.", "", "", "", {}, LIMITATIONS
        result = pipeline.analyze(
            image, measured_grams=_clean_measured(measured), top_k=int(top_k)
        )
        notes = "\n".join(f"- {w}" for w in result.warnings) or "No warnings."
        return (
            _classification_text(result),
            result.detection.summary() if result.detection else "not run",
            result.weight.summary(),
            _macros_text(result),
            analysis_to_dict(result),
            notes + "\n\n" + LIMITATIONS,
        )

    def run_multi_detect(image, top_k, max_dishes, crop_padding):
        return multi_detect(multi_pipeline, image, top_k, max_dishes, crop_padding)

    def run_multi_compute(state, table):
        return multi_compute(multi_pipeline, state, table)

    with gr.Blocks(title="Food Image Analysis") as demo:
        gr.Markdown(
            "# Food Image Analysis\n"
            "Upload a photo or take one with your camera, detect the dishes, then "
            "enter each dish's weight to get its macros and the plate total. Weight "
            "and totals are only shown for reliable data."
        )
        with gr.Tabs():
            with gr.Tab("Single dish"):
                _single_dish_tab(pipeline, run, gr)
            if multi_pipeline is not None:
                with gr.Tab("Multiple dishes"):
                    _multi_dish_tab(multi_pipeline, run_multi_detect, run_multi_compute, gr)
    return demo


def _single_dish_tab(pipeline: FoodAnalysisPipeline, run, gr):
    with gr.Row():
        with gr.Column():
            image_input = _image_input(gr, "Food image")
            measured_input = gr.Number(
                value=None, label="Measured weight (g, optional)", precision=1
            )
            top_k = gr.Slider(1, 5, value=int(pipeline.config.ui.max_top_k), step=1, label="Top-k")
            analyze_button = gr.Button("Analyze", variant="primary")
        with gr.Column():
            cls_out = gr.Textbox(label="Classification", lines=4)
            det_out = gr.Textbox(label="Detection", lines=2)
            weight_out = gr.Textbox(label="Portion weight", lines=2)
            macros_out = gr.Textbox(label="Macronutrients", lines=8)
            json_out = gr.JSON(label="Structured result")
            notes_out = gr.Markdown(value=LIMITATIONS)

    analyze_button.click(
        run,
        inputs=[image_input, measured_input, top_k],
        outputs=[cls_out, det_out, weight_out, macros_out, json_out, notes_out],
    )


def _multi_dish_tab(multi_pipeline: MultiDishPipeline, run_multi_detect, run_multi_compute, gr):
    det_cfg = multi_pipeline.config.detection
    state = gr.State(value=None)

    gr.Markdown(
        "**1)** Upload or capture a photo and click *Detect & classify*. "
        "**2)** In the table, edit each dish's **Grams (g)** and untick **Include** "
        "to leave a dish out, then click *Compute nutrition & total*."
    )
    with gr.Row():
        with gr.Column(scale=1):
            image_input = _image_input(gr, "Food image (upload or camera)")
            top_k = gr.Slider(
                1, 5, value=int(multi_pipeline.config.ui.max_top_k), step=1, label="Top-k per dish"
            )
            max_dishes = gr.Slider(1, 10, value=int(det_cfg.max_dishes), step=1, label="Max dishes")
            crop_padding = gr.Slider(
                0.0, 0.3, value=float(det_cfg.crop_padding), step=0.01, label="Crop padding"
            )
            detect_button = gr.Button("1) Detect & classify", variant="primary")
            compute_button = gr.Button("2) Compute nutrition & total")
        with gr.Column(scale=2):
            annotated_out = gr.Image(label="Detected dishes (numbered)", interactive=False)
            table_out = gr.Dataframe(
                headers=MULTI_HEADERS,
                datatype=MULTI_DATATYPE,
                interactive=True,
                label="Dishes — edit Include / Grams (g)",
            )
            report_out = gr.Textbox(label="Per-dish analysis & plate total", lines=18)
            json_out = gr.JSON(label="Structured result")
            notes_out = gr.Markdown(value=LIMITATIONS)

    detect_button.click(
        run_multi_detect,
        inputs=[image_input, top_k, max_dishes, crop_padding],
        outputs=[annotated_out, state, table_out, report_out, json_out, notes_out],
    )
    compute_button.click(
        run_multi_compute,
        inputs=[state, table_out],
        outputs=[report_out, json_out, notes_out],
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="FoodMacros Gradio app")
    parser.add_argument("--config", default=None, help="Path to a config JSON")
    parser.add_argument("--model-dir", default=None, help="Directory with classifier.keras")
    parser.add_argument("--regressor-dir", default=None, help="Directory with mass_regressor.keras")
    parser.add_argument("--share", action="store_true", help="Create a public share link")
    parser.add_argument("--port", type=int, default=None, help="Server port")
    args = parser.parse_args(argv)

    config = load_config(args.config) if args.config else load_config()
    if args.share:
        config.ui.share = True
    if args.port is not None:
        config.ui.server_port = args.port

    if not has_package("gradio"):
        print(
            "gradio is not installed. Install it with 'pip install gradio' to run the app.\n"
            "The rest of the project (classification, detection, weight, nutrition) does not "
            "require gradio."
        )
        return 1

    pipeline, resolved = create_pipeline(config, args.model_dir, args.regressor_dir)
    print(resolved.summary())
    for note in pipeline.notes:
        print(f"note: {note}")

    multi_pipeline = create_multi_pipeline(config, args.model_dir)
    for note in multi_pipeline.notes:
        print(f"note: {note}")

    demo = build_interface(pipeline, multi_pipeline=multi_pipeline)
    try:
        demo.launch(share=config.ui.share, server_port=config.ui.server_port)
    except Exception as exc:  # tunnel/network failures must not crash the shell
        print(f"Failed to launch Gradio: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
