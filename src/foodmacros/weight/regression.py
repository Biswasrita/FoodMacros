"""Nutrition5k-based mass & macronutrient regression (portion-weight source).

Predicts a dish's total mass (and fat/carb/protein/calories) from a top-down
image. This is the *evidence-based* weight route. It is only usable when a
Nutrition5k-style dataset (dish metadata + overhead RGB images) is attached;
otherwise every entry point reports "unavailable" with a clear reason.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Nutrition5k metadata targets (grams / kcal).
TARGET_COLUMNS = ["total_mass", "total_calories", "total_fat", "total_carb", "total_protein"]
REQUIRED_METADATA_COLUMNS = ["dish_id"] + TARGET_COLUMNS
DEFAULT_IMAGE_SUBDIR = "realsense_overhead"
DEFAULT_IMAGE_NAME = "rgb.png"

# Map model target -> human-readable nutrient name.
TARGET_TO_NUTRIENT = {
    "total_mass": "total_mass_g",
    "total_calories": "calories",
    "total_fat": "fat_g",
    "total_carb": "carb_g",
    "total_protein": "protein_g",
}


@dataclass
class MassPrediction:
    available: bool
    total_mass_g: Optional[float] = None
    calories: Optional[float] = None
    fat_g: Optional[float] = None
    carb_g: Optional[float] = None
    protein_g: Optional[float] = None
    reason: str = ""


@dataclass
class RegressionRecord:
    dish_id: str
    rgb_path: str
    targets: Dict[str, float] = field(default_factory=dict)


def find_metadata_files(root: str | Path) -> List[str]:
    root = Path(root)
    if not root.exists():
        return []
    patterns = [str(root / "dish_metadata*.csv"), str(root / "*" / "dish_metadata*.csv")]
    found: List[str] = []
    for pattern in patterns:
        found.extend(glob.glob(pattern))
    return sorted(set(found))


def load_dish_metadata(root: str | Path) -> pd.DataFrame:
    """Load and concatenate Nutrition5k dish-metadata CSVs from ``root``."""
    files = find_metadata_files(root)
    if not files:
        raise FileNotFoundError(
            f"No Nutrition5k metadata (dish_metadata*.csv) found under {root}."
        )
    frames = [pd.read_csv(f) for f in files]
    frame = pd.concat(frames, ignore_index=True)
    frame.columns = [str(c).strip() for c in frame.columns]
    missing = [c for c in REQUIRED_METADATA_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(
            f"Nutrition5k metadata is missing required column(s): {missing}. "
            f"Present: {list(frame.columns)}."
        )
    frame["dish_id"] = frame["dish_id"].astype(str).str.strip()
    for column in TARGET_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["dish_id"] + TARGET_COLUMNS).reset_index(drop=True)
    return frame


def build_regression_records(
    root: str | Path,
    metadata: Optional[pd.DataFrame] = None,
    image_subdir: str = DEFAULT_IMAGE_SUBDIR,
    image_name: str = DEFAULT_IMAGE_NAME,
) -> Tuple[List[RegressionRecord], int]:
    """Pair each dish with its overhead RGB image. Returns (records, skipped)."""
    root = Path(root)
    metadata = load_dish_metadata(root) if metadata is None else metadata
    records: List[RegressionRecord] = []
    skipped = 0
    for _, row in metadata.iterrows():
        image_path = root / image_subdir / str(row["dish_id"]) / image_name
        if not image_path.exists():
            skipped += 1
            continue
        records.append(
            RegressionRecord(
                dish_id=str(row["dish_id"]),
                rgb_path=str(image_path),
                targets={c: float(row[c]) for c in TARGET_COLUMNS},
            )
        )
    return records, skipped


def compute_norm_stats(records: Sequence[RegressionRecord]) -> Dict[str, Tuple[float, float]]:
    if not records:
        raise ValueError("records is empty; cannot compute normalisation stats.")
    stats: Dict[str, Tuple[float, float]] = {}
    for column in TARGET_COLUMNS:
        values = np.array([r.targets[column] for r in records], dtype=np.float64)
        mean = float(values.mean())
        std = float(values.std()) or 1.0
        stats[column] = (mean, std)
    return stats


def normalize_targets(targets: Dict[str, float], stats: Dict[str, Tuple[float, float]]) -> np.ndarray:
    return np.array(
        [(targets[c] - stats[c][0]) / stats[c][1] for c in TARGET_COLUMNS], dtype=np.float32
    )


def denormalize_targets(vector, stats: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    if len(vector) != len(TARGET_COLUMNS):
        raise ValueError(f"Expected {len(TARGET_COLUMNS)} targets, got {len(vector)}.")
    return {c: float(vector[i] * stats[c][1] + stats[c][0]) for i, c in enumerate(TARGET_COLUMNS)}


def build_regression_model(
    num_targets: int = len(TARGET_COLUMNS),
    img_size: int = 224,
    backbone: str = "mobilenetv2",
    weights: str | None = "imagenet",
):
    """A small CNN regressor outputting normalised targets."""
    import tensorflow as tf

    apps = tf.keras.applications
    kwargs = dict(include_top=False, weights=weights, input_shape=(img_size, img_size, 3))
    if backbone == "mobilenetv2":
        base = apps.MobileNetV2(**kwargs)
    elif backbone == "efficientnetb0":
        base = apps.EfficientNetB0(**kwargs)
    else:
        raise ValueError(f"Unsupported regression backbone '{backbone}'.")

    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="image")
    x = base(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_targets, name="targets")(x)
    model = tf.keras.Model(inputs, outputs, name=f"{backbone}_regressor")
    return model, base


def get_regression_preprocess_input(backbone: str = "mobilenetv2"):
    import tensorflow as tf

    if backbone == "mobilenetv2":
        return tf.keras.applications.mobilenet_v2.preprocess_input
    if backbone == "efficientnetb0":
        return tf.keras.applications.efficientnet.preprocess_input
    return lambda x: x


def dataset_from_records(
    records: Sequence[RegressionRecord],
    img_size: int,
    batch_size: int,
    stats: Dict[str, Tuple[float, float]],
    shuffle: bool = False,
    seed: int | None = None,
    preprocess_fn=None,
):
    """Streaming tf.data of (image, normalised target vector) from records."""
    import tensorflow as tf

    paths = [r.rgb_path for r in records]
    targets = np.stack([normalize_targets(r.targets, stats) for r in records], axis=0)
    ds = tf.data.Dataset.from_tensor_slices((paths, targets))

    def _load(path, target):
        raw = tf.io.read_file(path)
        image = tf.io.decode_image(raw, channels=3, expand_animations=False)
        image = tf.image.resize(image, (img_size, img_size))
        return tf.cast(image, tf.float32), target

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(min(len(paths), 10000), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    if preprocess_fn is not None:
        ds = ds.map(lambda x, y: (preprocess_fn(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def predict_mass(
    model,
    preprocess_fn,
    image_rgb,
    img_size: int,
    stats: Dict[str, Tuple[float, float]],
) -> MassPrediction:
    """Predict total mass + macros from an RGB uint8 array."""
    import tensorflow as tf

    array = np.asarray(image_rgb)
    if array.ndim != 3 or array.shape[-1] != 3:
        return MassPrediction(False, reason="image must be an HxWx3 RGB array.")
    if array.shape[0] != img_size or array.shape[1] != img_size:
        array = tf.image.resize(array, (img_size, img_size)).numpy()
    x = preprocess_fn(array.astype(np.float32)) if preprocess_fn is not None else array.astype(np.float32)
    vector = np.asarray(model.predict(x[None, ...], verbose=0))[0]
    nutrients = denormalize_targets(vector, stats)
    return MassPrediction(
        available=True,
        total_mass_g=nutrients["total_mass"],
        calories=nutrients["total_calories"],
        fat_g=nutrients["total_fat"],
        carb_g=nutrients["total_carb"],
        protein_g=nutrients["total_protein"],
        reason="",
    )


def train_mass_regressor(
    records: Sequence[RegressionRecord],
    output_dir,
    img_size: int = 224,
    batch_size: int = 16,
    epochs: int = 10,
    learning_rate: float = 1e-4,
    backbone: str = "mobilenetv2",
    val_fraction: float = 0.2,
    seed: int = 42,
    weights: str | None = "imagenet",
    verbose: int = 1,
):
    """Train the mass/macro regressor and save model + normalisation stats."""
    import tensorflow as tf

    if not records:
        raise ValueError("records is empty; nothing to train on.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(records))
    n_val = max(1, int(round(len(records) * val_fraction)))
    val_idx = set(order[:n_val].tolist())
    train_records = [r for i, r in enumerate(records) if i not in val_idx]
    val_records = [r for i, r in enumerate(records) if i in val_idx]
    if not train_records:
        train_records = list(records)

    stats = compute_norm_stats(train_records)
    preprocess_fn = get_regression_preprocess_input(backbone)
    train_ds = dataset_from_records(
        train_records, img_size, batch_size, stats, shuffle=True, seed=seed, preprocess_fn=preprocess_fn
    )
    val_ds = dataset_from_records(
        val_records, img_size, batch_size, stats, shuffle=False, preprocess_fn=preprocess_fn
    )

    model, base = build_regression_model(len(TARGET_COLUMNS), img_size, backbone, weights=weights)
    for layer in base.layers:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss="mse", metrics=["mae"])
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=verbose)

    model.save(output_dir / "mass_regressor.keras")
    (output_dir / "regression_norm_stats.json").write_text(
        json.dumps({"targets": TARGET_COLUMNS, "stats": stats}, indent=2), encoding="utf-8"
    )
    return {"model_path": str(output_dir / "mass_regressor.keras"), "history": dict(history.history)}
