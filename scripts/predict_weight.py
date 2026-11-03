#!/usr/bin/env python
"""Predict food weight from an RGB image using trained mass regressor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from foodmacros.data.preprocessing import load_image
from foodmacros.weight.regression import get_regression_preprocess_input


def predict_mass(image_path: str | Path, model_dir: str | Path = "models/weight"):
    import tensorflow as tf

    model_dir = Path(model_dir)
    model_path = model_dir / "mass_regressor.keras"
    if not model_path.exists():
        for alt in ["mass_only_regressor.keras", "regressor.keras"]:
            cand = model_dir / alt
            if cand.exists():
                model_path = cand
                break
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found in {model_dir}")

    norm_path = model_dir / "normalization.json"
    stats_path = model_dir / "regression_norm_stats.json"
    mean = None
    std = None
    img_size = 224
    backbone = "mobilenetv2"
    if norm_path.exists():
        raw = json.loads(norm_path.read_text())
        mean = float(raw.get("mean"))
        std = float(raw.get("std", 1.0)) or 1.0
        img_size = int(raw.get("img_size", 224))
        backbone = raw.get("backbone", "mobilenetv2")
    elif stats_path.exists():
        raw = json.loads(stats_path.read_text())
        if "stats" in raw and "total_mass" in raw["stats"]:
            mean, std = float(raw["stats"]["total_mass"][0]), float(raw["stats"]["total_mass"][1]) or 1.0
        else:
            # try first
            k = list(raw.get("stats", {}).keys())[0]
            mean, std = float(raw["stats"][k][0]), float(raw["stats"][k][1]) or 1.0
    if mean is None or std is None:
        raise ValueError("Normalization stats not found")

    model = tf.keras.models.load_model(model_path, compile=False)
    preprocess_fn = get_regression_preprocess_input(backbone)
    # load image
    try:
        img = load_image(image_path, img_size)
    except Exception:
        from PIL import Image

        im = Image.open(image_path).convert("RGB")
        im = im.resize((img_size, img_size))
        img = np.array(im).astype(np.float32)
    x = preprocess_fn(img.astype(np.float32)) if preprocess_fn is not None else img.astype(np.float32)
    pred = model.predict(x[None, ...], verbose=0)
    val = float(np.asarray(pred).reshape(-1)[0])
    mass_g = val * std + mean
    return float(mass_g)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Predict food weight from RGB image")
    parser.add_argument("--image", required=True, help="Path to food image")
    parser.add_argument("--model", default="models/weight", help="Model directory")
    args = parser.parse_args(argv)
    try:
        mass_g = predict_mass(args.image, args.model)
        print(f"Estimated food weight: {mass_g:.1f} g")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
