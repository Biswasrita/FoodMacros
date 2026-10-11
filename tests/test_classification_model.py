"""Stage 4 tests for model construction."""

import numpy as np
import pytest

pytest.importorskip("tensorflow")

from foodmacros.classification.model import (  # noqa: E402
    build_model,
    compile_model,
    get_preprocess_input,
    make_loss,
    set_backbone_trainable,
    unfreeze_backbone_fraction,
)


def test_build_model_output_shape_and_metadata():
    built = build_model(3, img_size=32, weights=None)
    assert built.model.output_shape == (None, 3)
    assert built.num_classes == 3
    assert "efficientnetb0" in built.backbones
    assert built.use_vit_fusion is False


def test_build_model_rejects_bad_num_classes():
    with pytest.raises(ValueError):
        build_model(1, img_size=32, weights=None)


def test_compile_model_with_and_without_smoothing():
    built = build_model(3, img_size=32, weights=None)
    compile_model(built, learning_rate=1e-3, label_smoothing=0.1)
    assert built.model.optimizer is not None


def test_make_loss_variants():
    import tensorflow as tf

    standard = make_loss(0.0)
    assert isinstance(standard, tf.keras.losses.SparseCategoricalCrossentropy)
    smoothed = make_loss(0.2)
    value = smoothed(
        np.array([0, 1]),
        np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]], dtype="float32"),
    )
    assert np.isfinite(np.asarray(value)).all()


def test_freeze_then_unfreeze_increases_trainable_variables():
    built = build_model(3, img_size=32, weights=None)
    set_backbone_trainable(built, False)
    frozen = len(built.model.trainable_variables)
    unfreeze_backbone_fraction(built, fraction=0.2)
    warmed = len(built.model.trainable_variables)
    assert warmed > frozen


def test_get_preprocess_input_is_callable():
    fn = get_preprocess_input("efficientnetb0")
    out = fn(np.zeros((2, 2, 3), dtype="float32"))
    assert out.shape == (2, 2, 3)
