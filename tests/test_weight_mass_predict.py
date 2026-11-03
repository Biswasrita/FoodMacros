"""Tests for mass-only prediction utilities."""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorflow")

from foodmacros.weight import regression


def test_build_mass_only_regressor_shape():
    model, base = regression.build_mass_only_regressor(img_size=32, weights=None)
    assert model.output_shape[-1] == 1


def test_predict_mass_only_from_array(tmp_path):
    model, _ = regression.build_mass_only_regressor(img_size=32, weights=None)
    from foodmacros.weight.regression import get_regression_preprocess_input

    img = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
    mass = regression.predict_mass_only_from_array(
        model, get_regression_preprocess_input("mobilenetv2"), img, 32, mean=200.0, std=50.0
    )
    assert isinstance(mass, float)
    assert np.isfinite(mass)
    assert mass > 0  # denorm gives reasonable positive


def test_load_regression_uses_saved_input_size(tmp_path):
    from foodmacros.config import load_config
    from foodmacros.pipeline import _load_regression

    img_size = 96
    model, _ = regression.build_mass_only_regressor(img_size=img_size, weights=None)
    model.save(tmp_path / "mass_regressor.keras")
    (tmp_path / "normalization.json").write_text(
        json.dumps(
            {
                "mode": "mass_only",
                "target": "mass_g",
                "mean": 250.0,
                "std": 50.0,
                "img_size": img_size,
                "backbone": "mobilenetv2",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"img_size": img_size, "backbone": "mobilenetv2"}),
        encoding="utf-8",
    )

    loaded = _load_regression(tmp_path, load_config())

    assert loaded is not None
    assert loaded["img_size"] == img_size  # from the weights, not the classifier config
    assert loaded["stats"] == {"total_mass": (250.0, 50.0)}
    prediction = regression.predict_mass(
        loaded["model"],
        loaded["preprocess_fn"],
        np.zeros((img_size, img_size, 3), dtype=np.uint8),
        loaded["img_size"],
        loaded["stats"],
    )
    assert prediction.available
    assert prediction.total_mass_g is not None
    assert np.isfinite(prediction.total_mass_g)


def test_resolve_regressor_dir_defaults_and_explicit(tmp_path, monkeypatch):
    import foodmacros.pipeline as pipeline

    monkeypatch.setattr(pipeline, "DEFAULT_REGRESSOR_DIR", tmp_path)
    assert pipeline._resolve_regressor_dir() is None

    (tmp_path / "mass_regressor.keras").write_bytes(b"x")
    assert pipeline._resolve_regressor_dir() == tmp_path
    assert pipeline._resolve_regressor_dir("explicit/dir") == Path("explicit/dir")

