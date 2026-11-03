#!/usr/bin/env python
"""Re-save a classifier ``.keras`` archive in the installed Keras format.

Older Keras releases cannot deserialize layer configs written by newer ones
(newer versions serialise ``quantization_config: None`` into ``Dense`` configs).
This tool loads such an archive through the controlled compatibility loader in
``foodmacros.classification.predict``, verifies that the weights/behaviour are
unchanged, and re-saves it so future loads are native.

Usage:
    python scripts/migrate_classifier.py --model models/classifier/classifier.keras
    python scripts/migrate_classifier.py --check          # report only, no write
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from foodmacros.classification.predict import _load_keras_model  # noqa: E402

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "classifier" / "classifier.keras"
DEFAULT_BACKUP_DIR = Path(__file__).resolve().parents[1] / "backups"


def _probe_inputs(input_shape) -> list:
    """Deterministic inputs covering black/white/mid-grey/noise images."""
    shape = tuple(input_shape)
    if len(shape) != 4:
        raise ValueError(f"Expected a 4-D model input, got {shape}.")
    batch, height, width, channels = (d or 1 for d in shape)
    rng = np.random.default_rng(0)
    probes = [
        np.zeros((batch, height, width, channels), dtype=np.float32),
        np.ones((batch, height, width, channels), dtype=np.float32) * 127.5,
        np.ones((batch, height, width, channels), dtype=np.float32) * 255.0,
        rng.uniform(0.0, 255.0, (batch, height, width, channels)).astype(np.float32),
    ]
    return probes


def _max_difference(model, other) -> float:
    largest = 0.0
    for probe in _probe_inputs(model.input_shape):
        a = np.asarray(model.predict(probe, verbose=0), dtype=np.float64)
        b = np.asarray(other.predict(probe, verbose=0), dtype=np.float64)
        if a.shape != b.shape:
            raise ValueError(f"Prediction shapes differ: {a.shape} vs {b.shape}.")
        largest = max(largest, float(np.max(np.abs(a - b))))
    return largest


def migrate(model_path: Path, check_only: bool = False, atol: float = 1e-5) -> int:
    import tensorflow as tf

    model_path = Path(model_path)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return 1

    print(f"TensorFlow {tf.__version__} | Keras {tf.keras.__version__}")
    print(f"Loading {model_path} ...")
    model = _load_keras_model(model_path)

    target = model_path.with_name(model_path.stem + ".migrated.keras")
    print(f"Saving with the installed Keras to {target.name} ...")
    model.save(target)

    reloaded = tf.keras.models.load_model(target, compile=False)
    print(f"Native reload: input={reloaded.input_shape} output={reloaded.output_shape}")

    difference = _max_difference(model, reloaded)
    print(f"Max |prediction difference| across probes: {difference:.3e}")
    if difference > atol:
        target.unlink(missing_ok=True)
        print("Migration aborted: predictions changed; original left untouched.")
        return 1

    if check_only:
        target.unlink(missing_ok=True)
        print("Check only: archive can be re-saved safely; original left untouched.")
        return 0

    backup_dir = DEFAULT_BACKUP_DIR
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / (model_path.name + ".pre-migration")
    shutil.copy2(model_path, backup)
    print(f"Backup written to {backup}")

    target.replace(model_path)
    print(f"Re-saved {model_path} in the installed Keras format.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to classifier.keras")
    parser.add_argument(
        "--check", action="store_true", help="Verify the migration without writing anything"
    )
    args = parser.parse_args(argv)
    return migrate(Path(args.model), check_only=args.check)


if __name__ == "__main__":
    sys.exit(main())
