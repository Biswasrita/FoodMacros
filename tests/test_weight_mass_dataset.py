"""Tests for RGB-only Nutrition5k mass dataset utilities."""

import os

import numpy as np
import pandas as pd
import pytest

from foodmacros.weight import mass_dataset


HEADER = "dish_id,total_mass,total_calories,total_fat,total_carb,total_protein"


@pytest.fixture
def nutrition5k_tree(tmp_path, image_factory):
    root = tmp_path / "nutrition5k"
    root.mkdir()
    rows = [
        "dish_1,320,450,12,55,25",
        "dish_2,180,210,6,20,9",
        "dish_3,500,600,20,70,30",  # no image -> skipped
    ]
    (root / "dish_metadata_cafe1.csv").write_text("\n".join([HEADER] + rows), encoding="utf-8")
    for dish in ("dish_1", "dish_2"):
        image_factory(root / "imagery" / "realsense_overhead" / dish / "rgb.png")
    return root


def test_find_metadata(nutrition5k_tree):
    files = mass_dataset.find_nutrition5k_metadata(nutrition5k_tree)
    assert len(files) >= 1


def test_load_metadata_both_cafes(tmp_path):
    root = tmp_path / "n5k"
    root.mkdir()
    (root / "dish_metadata_cafe1.csv").write_text("dish_id,total_mass\nA,100\n", encoding="utf-8")
    (root / "dish_metadata_cafe2.csv").write_text("dish_id,total_mass\nB,200\n", encoding="utf-8")
    df = mass_dataset.load_nutrition5k_mass_metadata(root)
    assert set(df["dish_id"]) == {"A", "B"}


def test_build_weight_dataset_skips_missing(nutrition5k_tree):
    df, stats = mass_dataset.build_weight_dataset(nutrition5k_tree)
    assert len(df) == 2
    assert stats["missing_images"] == 1
    assert set(df["dish_id"]) == {"dish_1", "dish_2"}
    assert all(df["mass_g"] > 0)


def test_build_weight_dataset_skips_missing_mass(tmp_path, image_factory):
    root = tmp_path / "n5k"
    root.mkdir()
    (root / "dish_metadata_cafe1.csv").write_text("dish_id,total_mass\nd1,\nd2,150\n", encoding="utf-8")
    image_factory(root / "imagery" / "realsense_overhead" / "d2" / "rgb.png")
    df, stats = mass_dataset.build_weight_dataset(root)
    assert len(df) == 1
    assert stats["missing_weights"] >= 1


def test_build_weight_dataset_skips_invalid_mass(tmp_path, image_factory):
    root = tmp_path / "n5k"
    root.mkdir()
    (root / "dish_metadata_cafe1.csv").write_text("dish_id,total_mass\nd1,0\nd2,-5\nd3,100\n", encoding="utf-8")
    image_factory(root / "imagery" / "realsense_overhead" / "d3" / "rgb.png")
    df, stats = mass_dataset.build_weight_dataset(root)
    assert len(df) == 1
    assert stats["invalid_weights"] == 2


def test_split_by_dish_id_no_leakage(tmp_path):
    df = pd.DataFrame(
        {
            "image_path": ["a", "b", "c", "d", "e"],
            "dish_id": ["d1", "d1", "d2", "d3", "d4"],  # d1 appears twice
            "mass_g": [100, 110, 200, 300, 400],
        }
    )
    train_df, val_df, test_df = mass_dataset.split_by_dish_id(df, 0.6, 0.2, 0.2, seed=123)
    # no dish overlap
    assert train_df["dish_id"].isin(val_df["dish_id"]).sum() == 0
    assert train_df["dish_id"].isin(test_df["dish_id"]).sum() == 0
    assert val_df["dish_id"].isin(test_df["dish_id"]).sum() == 0
