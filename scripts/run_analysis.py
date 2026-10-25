"""CLI: analyze a single food image end to end.

Usage:
    python scripts/run_analysis.py path/to/image.jpg [options]

Options:
    --measured G   measured portion weight in grams (the only "true" weight)
    --top-k N      number of class predictions to show
    --model-dir D  directory containing classifier.keras + class_names.json
    --regressor-dir D  directory containing mass_regressor.keras (Nutrition5k)
    --config P     path to a config JSON
    --json         print the structured JSON result instead of the text report

Missing models/sources are reported honestly; nothing is fabricated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from foodmacros.config import load_config  # noqa: E402
from foodmacros.pipeline import FoodAnalysisPipeline  # noqa: E402
from foodmacros.reporting import analysis_to_dict, format_analysis  # noqa: E402


def _apply_calibration_flags(config, args):
    weight = config.weight
    if args.cm_per_pixel is not None:
        weight.cm_per_pixel = args.cm_per_pixel
    if args.cm_per_depth_unit is not None:
        weight.cm_per_depth_unit = args.cm_per_depth_unit
    if args.reference_depth is not None:
        weight.reference_depth = args.reference_depth
    if args.invert_depth:
        weight.invert_depth = True
    if args.enable_depth_weight:
        weight.allow_experimental_depth = True
    return config


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Analyze one food image.")
    parser.add_argument("image", help="Path to the food image")
    parser.add_argument("--measured", type=float, default=None, help="Advanced override: measured weight (g)")
    parser.add_argument("--top-k", type=int, default=3, help="Top-k predictions")
    parser.add_argument("--model-dir", default=None, help="Classifier artifact directory")
    parser.add_argument("--regressor-dir", default=None, help="Mass regressor directory")
    parser.add_argument("--config", default=None, help="Path to a config JSON")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    parser.add_argument("--cm-per-pixel", type=float, default=None, help="Physical calibration: cm per pixel")
    parser.add_argument("--cm-per-depth-unit", type=float, default=None, help="Physical calibration: cm per relative depth unit")
    parser.add_argument("--reference-depth", type=float, default=None, help="Reference plane depth in MiDaS units")
    parser.add_argument("--invert-depth", action="store_true", help="Invert depth orientation if needed")
    parser.add_argument("--enable-depth-weight", action="store_true", help="Enable experimental depth weighting (requires calibration)")
    args = parser.parse_args(argv)

    if not Path(args.image).exists():
        print(f"Image not found: {args.image}")
        return 1

    config = load_config(args.config) if args.config else load_config()
    config = _apply_calibration_flags(config, args)
    pipeline, resolved = FoodAnalysisPipeline.from_discovery(
        config, model_dir=args.model_dir, mass_regressor_dir=args.regressor_dir
    )
    if not args.json:
        print(resolved.summary())
        for note in pipeline.notes:
            print(f"note: {note}")
        print()

    result = pipeline.analyze(args.image, measured_grams=args.measured, top_k=args.top_k)
    if args.json:
        print(json.dumps(analysis_to_dict(result), indent=2))
    else:
        print(format_analysis(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
