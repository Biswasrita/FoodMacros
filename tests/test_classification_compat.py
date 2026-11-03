"""Classifier loading across Keras serialization versions.

Newer Keras releases serialize ``quantization_config: None`` into layer configs.
Older releases reject that keyword, so ``load_classifier`` must be able to load
such archives without changing the model's behaviour.
"""

import json
import zipfile

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from foodmacros.classification.predict import (  # noqa: E402
    _strip_null_newer_keras_keys,
    load_classifier,
)
from foodmacros.data.splits import save_class_names  # noqa: E402


def _tiny_model(num_classes=3):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input((16, 16, 3)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model


def _inject_quantization_config(path):
    """Mimic Keras >= 3.13 by adding a null ``quantization_config`` to Dense."""
    with zipfile.ZipFile(path) as src:
        entries = {info.filename: src.read(info.filename) for info in src.infolist()}

    def inject(node):
        if isinstance(node, dict):
            if node.get("class_name") == "Dense" and isinstance(node.get("config"), dict):
                node["config"]["quantization_config"] = None
            for value in node.values():
                inject(value)
        elif isinstance(node, list):
            for value in node:
                inject(value)

    config = json.loads(entries["config.json"])
    inject(config)
    entries["config.json"] = json.dumps(config).encode("utf-8")

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as dst:
        for name, data in entries.items():
            dst.writestr(name, data)


def _save_bundle(tmp_path, model, names):
    model.save(tmp_path / "classifier.keras")
    save_class_names(tmp_path / "class_names.json", names)


def test_strip_removes_only_null_quantization_config():
    config = {
        "layers": [
            {"quantization_config": None, "units": 1},
            {"quantization_config": {"scheme": "int8"}, "units": 2},
            {"units": 3},
        ]
    }
    removed = _strip_null_newer_keras_keys(config)
    assert removed == 1
    assert "quantization_config" not in config["layers"][0]
    assert config["layers"][1]["quantization_config"] == {"scheme": "int8"}


def test_load_classifier_handles_newer_keras_serialization(tmp_path):
    names = ["a", "b", "c"]
    native = _tiny_model(len(names))
    _save_bundle(tmp_path, native, names)
    _inject_quantization_config(tmp_path / "classifier.keras")

    loaded, loaded_names = load_classifier(tmp_path)

    assert loaded_names == names
    probe = np.random.default_rng(0).uniform(0, 255, (16, 16, 3)).astype(np.float32)
    expected = np.asarray(native.predict(probe[None, ...], verbose=0))
    actual = np.asarray(loaded.predict(probe[None, ...], verbose=0))
    assert np.allclose(actual, expected, atol=1e-6)


def test_load_classifier_prefers_native_serialization(tmp_path):
    names = ["a", "b"]
    _save_bundle(tmp_path, _tiny_model(len(names)), names)
    loaded, loaded_names = load_classifier(tmp_path)
    assert loaded_names == names
    assert loaded.output_shape[-1] == len(names)
