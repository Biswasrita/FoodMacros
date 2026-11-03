"""Test to ensure dish-level split prevents leakage."""

import pandas as pd

from foodmacros.weight import mass_dataset


def test_no_dish_leakage_across_splits():
    # create synthetic data with repeated dish_ids
    rows = []
    for i in range(10):
        dish = f"d{i}"
        for j in range(2):  # multiple images per dish possible
            rows.append({"image_path": f"{dish}_{j}.png", "dish_id": dish, "mass_g": 100 + i * 10})
    df = pd.DataFrame(rows)
    train_df, val_df, test_df = mass_dataset.split_by_dish_id(df, 0.7, 0.15, 0.15, seed=42)
    train_dishes = set(train_df["dish_id"])
    val_dishes = set(val_df["dish_id"])
    test_dishes = set(test_df["dish_id"])
    assert len(train_dishes & val_dishes) == 0
    assert len(train_dishes & test_dishes) == 0
    assert len(val_dishes & test_dishes) == 0
