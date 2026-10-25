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
    lines = [f"{prediction.label} ({prediction.confidence:.1%})"]
    for name, prob in prediction.top_k:
        lines.append(f"  {name}: {prob:.1%}")
    return "\n".join(lines)


def _macros_text(result) -> str:
    if not result.nutrition.available:
        return f"Unavailable: {result.nutrition.reason}"
    if result.nutrition.basis == "per_reference":
        header = (
            f"{result.nutrition.label} — Nutrition (reference, per 100 g, "
            f"{result.nutrition.grams:g} g). This is NOT the nutrition for the detected portion."
        )
    else:
        header = f"{result.nutrition.label} — Nutrition for estimated/measured portion ({result.nutrition.grams:g} g) [{result.nutrition.basis}]"
    text = header + "\n" + format_nutrients(result.nutrition.nutrients)
    if result.nutrition.note:
        text += f"\nNote: {result.nutrition.note}"
    return text


def _clean_measured(measured):
    if measured is None or str(measured).strip() == "":
        return None
    try:
        candidate = float(measured)
    except (TypeError, ValueError):
        return None
    return candidate if candidate > 0 else None


MULTI_HEADERS = ["Dish #", "Detected by detector", "Identified as (Food-101)", "Confidence", "Include", "Weight (g)", "Weight status"]
MULTI_DATATYPE = ["number", "str", "str", "str", "bool", "number", "str"]


def _image_input(gr, label):
    """An image input with upload + camera, tolerant of older Gradio APIs."""
    try:
        return gr.Image(type="filepath", sources=["upload", "webcam"], label=label)
    except TypeError:
        try:
            return gr.Image(type="filepath", sources=["upload", "webcam"], label=label)
        except Exception:
            return gr.Image(type="filepath", label=label)


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "y", "x", "include", "checked")
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
        try:
            return [list(row) for row in table.values.tolist()]
        except Exception:
            pass
    if hasattr(table, "tolist"):
        try:
            return table.tolist()
        except Exception:
            pass
    try:
        return list(table)
    except Exception:
        return []


def multi_detect(multi_pipeline, image, top_k, max_dishes, crop_padding):
    """Detect + classify a plate and return (annotated image, state, rows, ...)."""
    if image is None:
        return None, None, [], "No image provided. Upload or take a photo to begin.", {}, LIMITATIONS
    try:
        result = multi_pipeline.analyze(
            image,
            top_k=int(top_k) if top_k is not None else int(multi_pipeline.config.ui.max_top_k),
            max_dishes=int(max_dishes) if max_dishes is not None else int(multi_pipeline.config.detection.max_dishes),
            crop_padding=float(crop_padding) if crop_padding is not None else float(multi_pipeline.config.detection.crop_padding),
        )
    except Exception as exc:
        return None, None, [], f"Analysis failed: {exc}", {}, LIMITATIONS
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
        w = dish.weight
        rows.append(
            [
                dish.index,
                det.label if det else "whole image (no separate box)",
                pred.label if (pred is not None and pred.available) else "",
                (
                    f"{pred.confidence:.1%}"
                    if (pred is not None and pred.available and pred.confidence is not None)
                    else ""
                ),
                True,
                float(w.grams) if (w is not None and w.available and w.grams is not None) else None,
                w.status if w is not None else "unavailable",
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
    """Read the dish table's Include column and recompute nutrition."""
    if state is None:
        return (
            "No detection result yet. Upload a photo and click 'Detect dishes' first.",
            {},
            LIMITATIONS,
        )
    rows = _table_rows(table)
    include_by_index = {}
    for row in rows:
        if not row:
            continue
        try:
            idx = int(float(row[0]))
        except (TypeError, ValueError, IndexError):
            continue
        include_by_index[idx] = _as_bool(row[4]) if len(row) > 4 else True

    per_dish_grams = [None for _ in state.dishes]
    include = [include_by_index.get(d.index, True) for d in state.dishes]
    try:
        updated = multi_pipeline.apply_weights(state, per_dish_grams, include)
    except Exception as exc:
        return f"Failed to compute nutrition: {exc}", {}, LIMITATIONS
    return format_multi_analysis(updated), multi_analysis_to_dict(updated), LIMITATIONS


def build_interface(pipeline: FoodAnalysisPipeline, multi_pipeline=None):
    import gradio as gr

    def run(image):
        if image is None:
            return "No image provided. Upload or take a photo to begin.", "", "", "", {}, LIMITATIONS
        try:
            result = pipeline.analyze(image, top_k=int(pipeline.config.ui.max_top_k))
        except Exception as exc:
            return f"Analysis failed: {exc}", "", "", "", {}, LIMITATIONS
        notes = "\n".join(f"- {w}" for w in result.warnings) or "No warnings."
        return (
            _classification_text(result),
            result.detection.summary() if result.detection else "not run",
            result.weight.summary(),
            _macros_text(result),
            analysis_to_dict(result),
            notes + "\n\n" + LIMITATIONS,
        )

    def run_multi_detect(image, top_k_adv, max_dishes_adv, crop_padding_adv):
        return multi_detect(multi_pipeline, image, top_k_adv, max_dishes_adv, crop_padding_adv)

    def run_multi_compute(state, table):
        return multi_compute(multi_pipeline, state, table)

    with gr.Blocks(title="Food Image Analysis") as demo:
        gr.Markdown(
            "# Food Image Analysis\n"
            "Upload a photo or take one with your camera. The app detects, classifies, "
            "and estimates nutrition automatically. Weights are only shown when a reliable "
            "source is available; otherwise nutrition is shown for the reference quantity (per 100 g) only."
        )
        with gr.Tabs():
            with gr.Tab("Single dish"):
                _single_dish_tab(pipeline, run, gr)
            if multi_pipeline is not None:
                with gr.Tab("Multiple dishes"):
                    _multi_dish_tab(multi_pipeline, run_multi_detect, run_multi_compute, gr)
            with gr.Tab("Calibration (Setup)"):
                _calibration_tab(gr)
    return demo


def _calibration_tab(gr):
    gr.Markdown(
        "## Calibration Setup (separate from prediction)\n\n"
        "**No fake calibration is created.** You must collect real measurements:\n\n"
        "1. **Photos with references** - Include a known-size object in the same plane as food "
        "(e.g., credit card 85.6×53.98 mm, ruler). Optionally include a known-height object on the same surface.\n"
        "2. **Actual weights** - Measure food on a digital kitchen scale in grams.\n"
        "3. **Record data** - Add samples to `data/calibration/calibration_samples.json` with image paths, "
        "reference dimensions/pixels, measured weights, and split (train/val/holdout).\n\n"
        "**Important:** MiDaS gives relative depth; a single reference is not guaranteed to be sufficient. "
        "Calibration is only used if fitted from real data and validated on held-out samples. "
        "Without real calibration data, weight estimation remains **unavailable** and nutrition is shown as **reference per 100 g**."
    )
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Load calibration samples")
            samples_path = gr.Textbox(label="Samples JSON path", value="data/calibration/calibration_samples.json")
            load_btn = gr.Button("Load & Validate Samples")
        with gr.Column():
            gr.Markdown("### Fit calibration from real measurements")
            fit_btn = gr.Button("Fit Calibration (train/val/holdout)")
    status_out = gr.Textbox(label="Status/Report", lines=12)
    report_json = gr.JSON(label="Details")

    def load_samples(path):
        try:
            from foodmacros.weight.calibration_dataset import load_calibration_samples, validate_sample
            samples = load_calibration_samples(path)
            if not samples:
                return "No samples found. Create data/calibration/calibration_samples.json with real measurements.", {}
            errors = []
            for s in samples:
                errs = validate_sample(s)
                for e in errs:
                    errors.append(f"{s.id}: {e}")
            msg = f"Loaded {len(samples)} samples.\n"
            if errors:
                msg += "Validation errors:\n" + "\n".join(errors[:20])
            else:
                msg += "All samples passed basic validation."
            return msg, {"count": len(samples), "errors": errors[:50]}
        except Exception as e:
            return f"Error: {e}", {}

    def fit_cal(path):
        try:
            from foodmacros.weight.calibration_dataset import load_calibration_samples
            from foodmacros.weight.calibration_fit import fit_calibration_from_samples, save_calibration
            samples = load_calibration_samples(path)
            if not samples:
                return "No samples to fit. Add real measurements first.", {}
            res = fit_calibration_from_samples(samples)
            if res.success and res.calibrated_params:
                try:
                    save_calibration(res.calibrated_params, res.report)
                    msg = f"Saved calibration to models/calibration/depth_calibration.json\n"
                except Exception as e:
                    msg = f"Fit completed but save failed: {e}\n"
            else:
                msg = ""
            msg += f"Success: {res.success}\n"
            msg += f"Errors: {res.errors}\n"
            msg += f"Warnings: {res.warnings}\n"
            msg += json.dumps(res.report, indent=2)
            return msg, res.report
        except Exception as e:
            return f"Error: {e}", {}

    load_btn.click(load_samples, inputs=[samples_path], outputs=[status_out, report_json])
    fit_btn.click(fit_cal, inputs=[samples_path], outputs=[status_out, report_json])


def _single_dish_tab(pipeline: FoodAnalysisPipeline, run, gr):
    with gr.Row():
        with gr.Column():
            image_input = _image_input(gr, "Upload or take a photo of your food")
            analyze_button = gr.Button("Analyze", variant="primary")
            with gr.Accordion("Advanced settings", open=False):
                top_k = gr.Slider(1, 5, value=int(pipeline.config.ui.max_top_k), step=1, label="Top-k (inspection only)")
        with gr.Column():
            cls_out = gr.Textbox(label="Classification (Food-101)", lines=4)
            det_out = gr.Textbox(label="Detection (separate taxonomy)", lines=2)
            weight_out = gr.Textbox(label="Portion weight", lines=2)
            macros_out = gr.Textbox(label="Nutrition (portion or reference)", lines=12)
            json_out = gr.JSON(label="Structured result")
            notes_out = gr.Markdown(value=LIMITATIONS)

    analyze_button.click(
        run,
        inputs=[image_input],
        outputs=[cls_out, det_out, weight_out, macros_out, json_out, notes_out],
    )


def _multi_dish_tab(multi_pipeline: MultiDishPipeline, run_multi_detect, run_multi_compute, gr):
    det_cfg = multi_pipeline.config.detection
    state = gr.State(value=None)

    gr.Markdown(
        "**Step 1:** Upload or capture a photo and click **Detect dishes**. "
        "**Step 2:** Review automatic weights and include/exclude dishes, then click "
        "**Compute nutrition & total**. "
        "Detector and classifier use different taxonomies (shown separately). "
        "A plate total appears only if every included dish has a valid weight and nutrition."
    )
    with gr.Row():
        with gr.Column(scale=1):
            image_input = _image_input(gr, "Upload or take a photo of your plate")
            detect_button = gr.Button("Detect dishes", variant="primary")
            compute_button = gr.Button("Compute nutrition & total")
            with gr.Accordion("Advanced settings", open=False):
                top_k = gr.Slider(1, 5, value=int(multi_pipeline.config.ui.max_top_k), step=1, label="Top-k per dish (inspection only)")
                max_dishes = gr.Slider(1, 10, value=int(det_cfg.max_dishes), step=1, label="Max dishes")
                crop_padding = gr.Slider(0.0, 0.3, value=float(det_cfg.crop_padding), step=0.01, label="Crop padding")
        with gr.Column(scale=2):
            annotated_out = gr.Image(label="Detected dishes (numbered)", interactive=False)
            table_out = gr.Dataframe(
                headers=MULTI_HEADERS,
                datatype=MULTI_DATATYPE,
                interactive=True,
                label="Dishes (include/exclude only; weights are automatic)",
            )
            report_out = gr.Textbox(label="Per-dish analysis & plate total", lines=20)
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

    try:
        pipeline, resolved = create_pipeline(config, args.model_dir, args.regressor_dir)
        print(resolved.summary())
        for note in pipeline.notes:
            print(f"note: {note}")

        multi_pipeline = create_multi_pipeline(config, args.model_dir)
        for note in multi_pipeline.notes:
            print(f"note: {note}")
    except Exception as exc:
        print(f"Failed to initialize pipelines: {exc}")
        return 1

    try:
        demo = build_interface(pipeline, multi_pipeline=multi_pipeline)
    except Exception as exc:
        print(f"Failed to build interface: {exc}")
        return 1
    try:
        demo.launch(share=config.ui.share, server_port=config.ui.server_port)
    except Exception as exc:  # tunnel/network failures must not crash the shell
        print(f"Failed to launch Gradio: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
