"""Stage 2 tests for image preprocessing and tf.data construction."""

import numpy as np
import pytest

from foodmacros.data.preprocessing import (
    ImageDecodeError,
    build_tf_dataset,
    is_decodable,
    load_image,
    preprocess_arrays,
    preprocess_image,
    safe_load_image,
)


def test_load_image_shape_and_dtype(synthetic_dataset):
    sample = next((synthetic_dataset["train"] / "pizza").iterdir())
    array = load_image(sample, img_size=48)
    assert array.shape == (48, 48, 3)
    assert array.dtype == np.uint8


def test_load_image_raises_on_invalid_file(tmp_path):
    bad = tmp_path / "not_an_image.png"
    bad.write_bytes(b"not really an image")
    with pytest.raises(ImageDecodeError):
        load_image(bad, img_size=32)
    assert safe_load_image(bad, img_size=32) is None
    assert is_decodable(bad) is False


def test_is_decodable_true_for_real_image(synthetic_dataset):
    sample = next((synthetic_dataset["train"] / "pizza").iterdir())
    assert is_decodable(sample) is True


def test_preprocess_image_batches_and_casts(synthetic_dataset):
    sample = next((synthetic_dataset["train"] / "pizza").iterdir())
    batch = preprocess_image(sample, img_size=40)
    assert batch.shape == (1, 40, 40, 3)
    assert batch.dtype == np.float32


def test_preprocess_image_applies_callable(synthetic_dataset):
    sample = next((synthetic_dataset["train"] / "pizza").iterdir())
    batch = preprocess_image(sample, img_size=32, preprocess_fn=lambda x: x / 255.0)
    assert batch.max() <= 1.0


def test_preprocess_arrays_stacks():
    arrays = [np.zeros((8, 8, 3), np.uint8), np.full((8, 8, 3), 255, np.uint8)]
    batch = preprocess_arrays(arrays)
    assert batch.shape == (2, 8, 8, 3)
    assert batch.dtype == np.float32


def test_build_tf_dataset_class_order(synthetic_dataset):
    tf = pytest.importorskip("tensorflow")
    names = ["pizza", "apple_pie", "bibimbap"]  # deliberately non-alphabetical
    ds = build_tf_dataset(
        synthetic_dataset["train"], names, img_size=32, batch_size=2, shuffle=False
    )
    assert ds.element_spec[1].dtype == tf.int32
    images, labels = next(iter(ds))
    assert images.shape == (2, 32, 32, 3)
    assert int(labels.numpy().max()) <= len(names) - 1


def test_build_tf_dataset_rejects_class_mismatch(synthetic_dataset):
    pytest.importorskip("tensorflow")
    with pytest.raises(ValueError, match="do not match class_names"):
        build_tf_dataset(
            synthetic_dataset["train"], ["pizza", "not_a_real_class"],
            img_size=32, batch_size=2,
        )
