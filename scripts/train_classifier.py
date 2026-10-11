"""CLI: train the food classifier on the discovered images.

Usage:
    python scripts/train_classifier.py [--config configs/default.json] [--full]

By default the config's ``runtime.mode`` (quick) is honoured. ``--full`` sets
mode to "full". Artifacts are written to ``runtime.output_dir`` (or the
runtime-resolved working dir) and can be loaded by the pipeline/Gradio app.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from foodmacros.classification.train import train_classifier  # noqa: E402
from foodmacros.config import load_config  # noqa: E402
from foodmacros.data.discovery import resolve_inputs  # noqa: E402
from foodmacros.data.splits import build_class_names, make_splits, save_class_names  # noqa: E402
from foodmacros.env_checks import default_working_dir  # noqa: E402


def _run_dir(config) -> Path:
    return Path(config.runtime.output_dir) if config.runtime.output_dir else default_working_dir()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train the food classifier.")
    parser.add_argument("--config", default=None, help="Path to a config JSON")
    parser.add_argument("--full", action="store_true", help="Force a full training run")
    args = parser.parse_args(argv)

    config = load_config(args.config) if args.config else load_config()
    if args.full:
        config.runtime.mode = "full"

    resolved = resolve_inputs(config)
    print(resolved.summary())
    if not resolved.image_root:
        print("\nNo image dataset found; attach 'hari31416/food-101' or set config.data.image_root.")
        return 1

    run_dir = _run_dir(config)
    class_names = build_class_names(
        resolved.image_root, resolved.validation_root, config.data.class_names_file
    )
    save_class_names(config.data.class_names_file or (run_dir / "class_names.json"), class_names)

    splits = make_splits(
        resolved.image_root,
        class_names,
        validation_root=resolved.validation_root,
        val_fraction=config.splits.val_fraction,
        seed=config.splits.shuffle_seed,
        test_source=config.splits.test_source,
    )
    print(splits.summary())

    result = train_classifier(
        config, class_names, splits.train, splits.val, output_dir=run_dir
    )
    print("\n" + result.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
