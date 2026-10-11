"""Stage 4 tests for single-image prediction (uses a tiny stand-in model)."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from foodmacros.classification.predict import predict_array, predict_path  # noqa: E402


def tiny_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input((32, 32, 3)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(3, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    return model


def test_predict_array_returns_ranked_topk():
    image = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
    result = predict_array(tiny_model(), None, image, ["a", "b", "c"], 32, top_k=3)
    assert result.available
    assert len(result.top_k) == 3
    assert result.label == result.top_k[0][0]
    probs = [p for _, p in result.top_k]
    assert probs == sorted(probs, reverse=True)
    assert abs(sum(probs) - 1.0) < 1e-3


def test_predict_array_top_k_is_clamped():
    image = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
    result = predict_array(tiny_model(), None, image, ["a", "b", "c"], 32, top_k=99)
    assert len(result.top_k) == 3


def test_predict_array_none_is_unavailable():
    result = predict_array(tiny_model(), None, None, ["a", "b", "c"], 32)
    assert not result.available
    assert "no image" in result.reason


def test_predict_array_wrong_shape_is_unavailable():
    result = predict_array(tiny_model(), None, np.zeros((10, 10), np.uint8), ["a", "b"], 32)
    assert not result.available
    assert "HxWx3" in result.reason


def test_predict_path_invalid_file_is_unavailable(tmp_path):
    bad = tmp_path / "broken.png"
    bad.write_bytes(b"not an image")
    result = predict_path(tiny_model(), None, bad, ["a", "b", "c"], 32)
    assert not result.available


def test_predict_path_real_image_is_available(synthetic_dataset):
    sample = next((synthetic_dataset["train"] / "pizza").iterdir())
    result = predict_path(tiny_model(), None, sample, ["a", "b", "c"], 32, top_k=2)
    assert result.available
    assert len(result.top_k) == 2
