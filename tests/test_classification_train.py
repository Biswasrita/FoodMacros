"""Stage 4 end-to-end smoke test: build -> train (2 quick steps) -> save -> reload.

Uses ``weights=None`` (no downloads) and a 32x32 input so it runs on CPU in
seconds. It validates the *plumbing*, not final accuracy.
"""

import json

import pytest

pytest.importorskip("tensorflow")

from foodmacros.classification.predict import load_classifier, predict_path  # noqa: E402
from foodmacros.classification.train import train_classifier  # noqa: E402
from foodmacros.config import AppConfig  # noqa: E402
from foodmacros.data.splits import build_class_names, make_splits  # noqa: E402


@pytest.fixture
def quick_config():
    config = AppConfig()
    config.runtime.mode = "quick"
    config.model.img_size = 32
    config.training.batch_size = 2
    config.training.quick_epochs = 1
    config.training.quick_steps_per_epoch = 1
    return config


def test_train_classifier_smoke_and_reload(synthetic_dataset, tmp_path, quick_config):
    class_names = build_class_names(synthetic_dataset["train"], synthetic_dataset["val"])
    splits = make_splits(
        synthetic_dataset["train"], class_names, synthetic_dataset["val"], seed=42
    )
    out = tmp_path / "run"

    result = train_classifier(
        quick_config,
        class_names,
        splits.train,
        splits.val,
        output_dir=out,
        weights=None,
        verbose=0,
    )

    assert (out / "classifier.keras").exists()
    assert (out / "class_names.json").exists()
    assert (out / "label_mapping.json").exists()

    mapping = json.loads((out / "label_mapping.json").read_text(encoding="utf-8"))
    assert mapping["0"] == class_names[0]

    model, reloaded_names = load_classifier(out)
    assert reloaded_names == class_names

    sample = next((synthetic_dataset["val"] / "pizza").iterdir())
    prediction = predict_path(model, None, sample, reloaded_names, 32, top_k=3)
    assert prediction.available
    assert len(prediction.top_k) == 3
