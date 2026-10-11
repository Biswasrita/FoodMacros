"""Stage 2 tests for deterministic, leak-free splitting."""

import pytest

from foodmacros.data.splits import (
    build_class_names,
    list_classes,
    load_class_names,
    make_splits,
    save_class_names,
)


def test_list_classes_is_sorted(synthetic_dataset):
    assert list_classes(synthetic_dataset["train"]) == sorted(synthetic_dataset["classes"])


def test_build_class_names_from_train(synthetic_dataset):
    names = build_class_names(synthetic_dataset["train"], synthetic_dataset["val"])
    assert names == sorted(synthetic_dataset["classes"])


def test_build_class_names_rejects_unexpected_val_class(synthetic_dataset, tmp_path, image_factory):
    extra_root = tmp_path / "val2"
    image_factory(extra_root / "unknown_class" / "x.png")
    with pytest.raises(ValueError, match="absent from the training class list"):
        build_class_names(synthetic_dataset["train"], extra_root)


def test_make_splits_is_deterministic(synthetic_dataset):
    names = build_class_names(synthetic_dataset["train"])
    a = make_splits(synthetic_dataset["train"], names, synthetic_dataset["val"], seed=42)
    b = make_splits(synthetic_dataset["train"], names, synthetic_dataset["val"], seed=42)
    assert a.train == b.train
    assert a.val == b.val
    assert a.test == b.test


def test_make_splits_no_leakage(synthetic_dataset):
    names = build_class_names(synthetic_dataset["train"])
    result = make_splits(synthetic_dataset["train"], names, synthetic_dataset["val"])
    train = {p for p, _ in result.train}
    val = {p for p, _ in result.val}
    test = {p for p, _ in result.test}
    assert not (train & val)
    assert not (train & test)
    assert not (val & test)
    assert len(result.train) + len(result.val) == 18  # all training images used
    assert len(result.test) == 6  # official validation root


def test_label_indices_match_class_names(synthetic_dataset):
    names = build_class_names(synthetic_dataset["train"])
    result = make_splits(synthetic_dataset["train"], names, synthetic_dataset["val"])
    for path, idx in result.train + result.val + result.test:
        class_from_path = path.replace("\\", "/").split("/")[-2]
        assert names[idx] == class_from_path


def test_carve_test_when_no_validation_root(synthetic_dataset):
    names = build_class_names(synthetic_dataset["train"])
    result = make_splits(synthetic_dataset["train"], names, validation_root=None)
    assert len(result.test) > 0
    assert any("carved from training" in n for n in result.notes)


def test_save_and_load_class_names(tmp_path):
    names = ["a", "b", "c"]
    path = save_class_names(tmp_path / "class_names.json", names)
    assert load_class_names(path) == names
