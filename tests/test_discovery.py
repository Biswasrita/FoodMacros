"""Stage 2 tests for input discovery."""

from pathlib import Path

from foodmacros.config import AppConfig
from foodmacros.data.discovery import (
    discover_detection_weights,
    discover_nutrition5k_roots,
    discover_nutrition_csvs,
    find_classification_roots,
    inspect_csv_columns,
    resolve_inputs,
)


def test_find_classification_roots(synthetic_dataset):
    roots = find_classification_roots([synthetic_dataset["root"]])
    paths = {r["root"] for r in roots}
    assert str(synthetic_dataset["train"]) in paths
    assert str(synthetic_dataset["val"]) in paths
    train_entry = next(r for r in roots if r["root"] == str(synthetic_dataset["train"]))
    assert train_entry["num_classes"] == 3
    assert train_entry["num_images"] == 18


def test_inspect_csv_columns(synthetic_nutrition_csv):
    cols = inspect_csv_columns(synthetic_nutrition_csv)
    assert "label" in cols and "weight" in cols and "calories" in cols


def test_discover_nutrition_csvs_validates_header(synthetic_nutrition_csv):
    results = discover_nutrition_csvs([synthetic_nutrition_csv.parent])
    assert len(results) == 1
    assert Path(results[0]["path"]) == synthetic_nutrition_csv
    assert "sodium" in {c.lower() for c in results[0]["columns"]}


def test_nutrition_csv_without_required_columns_is_ignored(tmp_path):
    bad = tmp_path / "other.csv"
    bad.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    assert discover_nutrition_csvs([tmp_path]) == []


def test_discover_detection_weights(tmp_path):
    (tmp_path / "yolo_food.pt").write_bytes(b"stub")
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    found = discover_detection_weights([tmp_path])
    assert len(found) == 1 and found[0].endswith("yolo_food.pt")


def test_discover_nutrition5k_roots(tmp_path):
    n5k = tmp_path / "nutrition5k"
    n5k.mkdir()
    (n5k / "dish_metadata_cafe1.csv").write_text("dish_id\n1\n", encoding="utf-8")
    found = discover_nutrition5k_roots([tmp_path])
    assert str(n5k) in found


def test_resolve_inputs_reports_missing_honestly(tmp_path):
    resolved = resolve_inputs(AppConfig(), search_roots=[tmp_path])
    assert resolved.image_root is None
    assert resolved.nutrition_csv is None
    assert any("Food image dataset not found" in m for m in resolved.missing)
    assert any("Nutrition CSV not found" in m for m in resolved.missing)
    assert any("Nutrition5k not found" in n for n in resolved.notes)
    assert any("detection weights" in n for n in resolved.notes)


def test_resolve_inputs_discovers_everything(synthetic_dataset, synthetic_nutrition_csv):
    (synthetic_dataset["root"] / "food_yolo.pt").write_bytes(b"stub")
    resolved = resolve_inputs(
        AppConfig(),
        search_roots=[synthetic_dataset["root"], synthetic_nutrition_csv.parent],
    )
    assert resolved.image_root == str(synthetic_dataset["train"])
    assert resolved.validation_root == str(synthetic_dataset["val"])
    assert resolved.nutrition_csv == str(synthetic_nutrition_csv)
    assert resolved.detection_weights is not None
    assert resolved.missing == []


def test_explicit_config_override_wins(synthetic_dataset, tmp_path):
    config = AppConfig()
    config.data.image_root = str(synthetic_dataset["train"])
    resolved = resolve_inputs(config, search_roots=[tmp_path])
    assert resolved.image_root == str(synthetic_dataset["train"])
