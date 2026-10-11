"""CLI: analyze a photo containing several dishes end to end.

Usage:
    python scripts/run_multi_analysis.py path/to/plate.jpg [options]

Options:
    --measured G         measured weight (g) of the whole plate; used only when
                         exactly one dish is detected (cannot be split)
    --dish-grams "G,G,.."  per-dish weights in grams, in detected order; use an
                         empty entry to leave a dish unweighed (e.g. "120,,80")
    --top-k N            number of class predictions to show per dish
    --max-dishes N       cap on the number of detected dishes analyzed
    --crop-padding F     fractional padding added around each detected box
    --model-dir D        directory containing classifier.keras + class_names.json
    --detector-weights P path to the YOLO food-detection .pt
    --nutrition-csv P    path to the nutrition CSV
    --config P           path to a config JSON
    --json               print the structured JSON result instead of text

By default the project's bundled artifacts are used
(models/classifier, models/food_detector, data/nutrition.csv). Missing
models/sources are reported honestly; nothing is fabricated.
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
from foodmacros.multi_pipeline import MultiDishPipeline  # noqa: E402
from foodmacros.multi_reporting import (  # noqa: E402
    format_multi_analysis,
    multi_analysis_to_dict,
)


def _parse_dish_grams(text):
    """Parse "120,,80" into [120.0, None, 80.0]; None for empty/blank parts."""
    if text is None:
        return None
    grams = []
    for part in str(text).split(","):
        part = part.strip()
        if part == "":
            grams.append(None)
            continue
        try:
            grams.append(float(part))
        except ValueError:
            grams.append(None)
    return grams


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Analyze a multi-dish food photo.")
    parser.add_argument("image", help="Path to the food image")
    parser.add_argument("--measured", type=float, default=None, help="Measured plate weight (g)")
    parser.add_argument(
        "--dish-grams",
        default=None,
        help='Per-dish weights in grams, e.g. --dish-grams "120,80,60"',
    )
    parser.add_argument("--top-k", type=int, default=3, help="Top-k predictions per dish")
    parser.add_argument("--max-dishes", type=int, default=None, help="Max dishes to analyze")
    parser.add_argument("--crop-padding", type=float, default=None, help="Fractional crop padding")
    parser.add_argument("--model-dir", default=None, help="Classifier artifact directory")
    parser.add_argument("--detector-weights", default=None, help="YOLO food-detector .pt")
    parser.add_argument("--nutrition-csv", default=None, help="Nutrition CSV path")
    parser.add_argument("--config", default=None, help="Path to a config JSON")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    args = parser.parse_args(argv)

    if not Path(args.image).exists():
        print(f"Image not found: {args.image}")
        return 1

    config = load_config(args.config) if args.config else load_config()
    pipeline = MultiDishPipeline.from_defaults(
        config,
        model_dir=args.model_dir,
        detector_weights=args.detector_weights,
        nutrition_csv=args.nutrition_csv,
    )

    if not args.json:
        for note in pipeline.notes:
            print(f"note: {note}")
        print()

    result = pipeline.analyze(
        args.image,
        top_k=args.top_k,
        measured_grams=args.measured,
        per_dish_grams=_parse_dish_grams(args.dish_grams),
        max_dishes=args.max_dishes,
        crop_padding=args.crop_padding,
    )
    if args.json:
        print(json.dumps(multi_analysis_to_dict(result), indent=2))
    else:
        print(format_multi_analysis(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
