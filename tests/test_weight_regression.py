"""Stage 5 tests for the Nutrition5k regression module."""

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tensorflow")

from foodmacros.weight.regression import (  # noqa: E402
    TARGET_COLUMNS,
    build_regression_model,
    build_regression_records,
    compute_norm_stats,
    dataset_from_records,
    denormalize_targets,
    find_metadata_files,
    load_dish_metadata,
    normalize_targets,
    predict_mass,
    train_mass_regressor,
)

HEADER = "dish_id,total_mass,total_calories,total_fat,total_carb,total_protein"
ROWS = [
    "dish_1,320,450,12,55,25",
    "dish_2,180,210,6,20,9",
    "dish_3,500,600,20,70,30",  # no image -> skipped
]


@pytest.fixture
def nutrition5k_tree(tmp_path, image_factory):
    root = tmp_path / "nutrition5k"
    root.mkdir()
    (root / "dish_metadata_cafe1.csv").write_text(
        "\n".join([HEADER] + ROWS), encoding="utf-8"
    )
    for dish in ("dish_1", "dish_2"):
        image_factory(root / "realsense_overhead" / dish / "rgb.png")
    return root


def test_find_and_load_metadata(nutrition5k_tree):
    assert find_metadata_files(nutrition5k_tree)
    metadata = load_dish_metadata(nutrition5k_tree)
    assert len(metadata) == 3
    assert set(TARGET_COLUMNS).issubset(metadata.columns)


def test_load_metadata_missing_column_raises(tmp_path):
    bad = tmp_path / "dish_metadata_bad.csv"
    bad.write_text("dish_id,total_mass\nx,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required column"):
        load_dish_metadata(tmp_path)


def test_build_records_skips_missing_images(nutrition5k_tree):
    records, skipped = build_regression_records(nutrition5k_tree)
    assert len(records) == 2
    assert skipped == 1
    assert records[0].targets["total_mass"] == pytest.approx(320.0)


def test_norm_round_trip(nutrition5k_tree):
    records, _ = build_regression_records(nutrition5k_tree)
    stats = compute_norm_stats(records)
    original = records[0].targets
    vector = normalize_targets(original, stats)
    restored = denormalize_targets(vector, stats)
    for column in TARGET_COLUMNS:
        assert restored[column] == pytest.approx(original[column], rel=1e-5)


def test_build_regression_model_shape():
    model, base = build_regression_model(len(TARGET_COLUMNS), img_size=32, weights=None)
    assert model.output_shape == (None, len(TARGET_COLUMNS))


def test_dataset_from_records_batches(nutrition5k_tree):
    records, _ = build_regression_records(nutrition5k_tree)
    stats = compute_norm_stats(records)
    ds = dataset_from_records(records, img_size=32, batch_size=2, stats=stats)
    images, targets = next(iter(ds))
    assert images.shape == (2, 32, 32, 3)
    assert targets.shape == (2, len(TARGET_COLUMNS))


def test_predict_mass_returns_prediction(nutrition5k_tree):
    records, _ = build_regression_records(nutrition5k_tree)
    stats = compute_norm_stats(records)
    model, _ = build_regression_model(len(TARGET_COLUMNS), img_size=32, weights=None)
    image = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
    from foodmacros.weight.regression import get_regression_preprocess_input

    prediction = predict_mass(model, get_regression_preprocess_input("mobilenetv2"), image, 32, stats)
    assert prediction.available
    assert np.isfinite(prediction.total_mass_g)


def test_train_mass_regressor_smoke(nutrition5k_tree, tmp_path):
    records, _ = build_regression_records(nutrition5k_tree)
    out = tmp_path / "reg"
    result = train_mass_regressor(
        records, out, img_size=32, batch_size=2, epochs=1, weights=None, verbose=0
    )
    assert (out / "mass_regressor.keras").exists()
    stats = json.loads((out / "regression_norm_stats.json").read_text(encoding="utf-8"))
    assert stats["targets"] == TARGET_COLUMNS
