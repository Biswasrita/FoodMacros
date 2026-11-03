"""RGB-only mass dataset utilities for Nutrition5k.

Builds image/weight pairs using only rgb.png and total_mass from
dish_metadata_cafe1.csv / dish_metadata_cafe2.csv, matched by dish_id.
"""

from __future__ import annotations

import glob
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_IMAGE_SUBDIR = "realsense_overhead"
DEFAULT_IMAGE_NAME = "rgb.png"


@dataclass
class MassDatasetRecord:
    image_path: str
    dish_id: str
    mass_g: float


def find_nutrition5k_metadata(nutrition5k_root: str | Path) -> List[Path]:
    root = Path(nutrition5k_root)
    if not root.exists():
        return []
    candidates = []
    # common patterns
    candidates.extend(root.glob("dish_metadata*.csv"))
    candidates.extend(root.glob("*/dish_metadata*.csv"))
    candidates.extend(root.glob("*/*/dish_metadata*.csv"))
    candidates.extend(root.glob("metadata/dish_metadata*.csv"))
    candidates.extend(root.glob("*/metadata/dish_metadata*.csv"))
    # rglob for cafe files
    for name in ["dish_metadata_cafe1.csv", "dish_metadata_cafe2.csv"]:
        for p in root.rglob(name):
            candidates.append(p)
    seen = set()
    found = []
    for c in candidates:
        try:
            rp = Path(c).resolve()
        except Exception:
            rp = Path(c)
        key = str(rp).lower()
        if key in seen:
            continue
        seen.add(key)
        found.append(Path(c))
    return sorted(set(found))


def load_nutrition5k_mass_metadata(nutrition5k_root: str | Path) -> pd.DataFrame:
    root = Path(nutrition5k_root)
    files = find_nutrition5k_metadata(root)
    if not files:
        raise FileNotFoundError(
            f"No Nutrition5k metadata (dish_metadata*.csv) found under {root}."
        )
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            frames.append(df)
        except Exception:
            continue
    if not frames:
        raise FileNotFoundError(f"No readable Nutrition5k metadata files found under {root}.")
    frame = pd.concat(frames, ignore_index=True)
    frame.columns = [str(c).strip() for c in frame.columns]
    if "dish_id" not in frame.columns:
        raise ValueError("Metadata missing dish_id column")
    mass_col = None
    for c in frame.columns:
        cl = str(c).lower()
        if cl == "total_mass" or cl == "mass" or cl == "total mass" or cl == "weight_g" or cl == "weight":
            mass_col = c
            break
    if mass_col is None:
        raise ValueError("Metadata missing total_mass column")
    frame = frame.rename(columns={mass_col: "total_mass"})
    frame["dish_id"] = frame["dish_id"].astype(str).str.strip()
    frame["total_mass"] = pd.to_numeric(frame["total_mass"], errors="coerce")
    # Keep rows without a mass so ``build_weight_dataset`` can report them as
    # ``missing_weights`` instead of silently dropping metadata rows. When a
    # dish appears more than once, prefer the row that actually has a mass.
    frame = frame.dropna(subset=["dish_id"])
    has_mass = frame["total_mass"].notna()
    frame = pd.concat([frame[has_mass], frame[~has_mass]])
    frame = frame.drop_duplicates(subset=["dish_id"], keep="first")
    return frame


def build_weight_dataset(
    nutrition5k_root: str | Path,
    output_csv: str | Path | None = None,
    image_subdir: str = DEFAULT_IMAGE_SUBDIR,
    image_name: str = DEFAULT_IMAGE_NAME,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    root = Path(nutrition5k_root)
    metadata = load_nutrition5k_mass_metadata(root)
    records: List[Dict[str, object]] = []
    missing_images = 0
    missing_weights = 0
    invalid_weights = 0

    patterns = [
        str(root / "imagery" / image_subdir / "*" / image_name),
        str(root / image_subdir / "*" / image_name),
        str(root / "*" / "*" / image_name),
    ]
    all_rgb = set()
    for pat in patterns:
        all_rgb |= set(glob.glob(pat))
    found_rgb = len(all_rgb)

    for _, row in metadata.iterrows():
        dish_id = str(row["dish_id"]).strip()
        mass = row.get("total_mass")
        if pd.isna(mass):
            missing_weights += 1
            continue
        try:
            mass_f = float(mass)
        except Exception:
            missing_weights += 1
            continue
        if mass_f <= 0 or not np.isfinite(mass_f):
            invalid_weights += 1
            continue
        img_path = None
        candidates = [
            root / "imagery" / image_subdir / dish_id / image_name,
            root / image_subdir / dish_id / image_name,
            root / dish_id / image_name,
        ]
        for c in candidates:
            if c.exists():
                img_path = c
                break
        if img_path is None:
            missing_images += 1
            continue
        try:
            img_resolved = str(img_path.resolve())
        except Exception:
            img_resolved = str(img_path)
        records.append(
            {
                "image_path": img_resolved,
                "dish_id": dish_id,
                "mass_g": float(mass_f),
            }
        )

    df = pd.DataFrame(records)
    if output_csv is not None:
        out = Path(output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
    stats = {
        "metadata_dishes": int(len(metadata)),
        "rgb_images_found": int(found_rgb),
        "valid_pairs": int(len(df)),
        "missing_images": int(missing_images),
        "missing_weights": int(missing_weights),
        "invalid_weights": int(invalid_weights),
    }
    return df, stats


# backward compat alias
build_weight_dataset_csv = build_weight_dataset


def split_by_dish_id(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("fractions must sum to 1.0")
    dish_ids = sorted(df["dish_id"].unique().tolist())
    rng = random.Random(seed)
    rng.shuffle(dish_ids)
    n = len(dish_ids)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    train_ids = set(dish_ids[:n_train])
    val_ids = set(dish_ids[n_train : n_train + n_val])
    test_ids = set(dish_ids[n_train + n_val :])
    train_df = df[df["dish_id"].isin(train_ids)].copy()
    val_df = df[df["dish_id"].isin(val_ids)].copy()
    test_df = df[df["dish_id"].isin(test_ids)].copy()
    # verify no leakage
    assert train_df["dish_id"].isin(val_df["dish_id"]).sum() == 0
    assert train_df["dish_id"].isin(test_df["dish_id"]).sum() == 0
    assert val_df["dish_id"].isin(test_df["dish_id"]).sum() == 0
    return train_df, val_df, test_df
