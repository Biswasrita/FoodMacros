"""Pytest configuration and shared fixtures for the FoodMacros test suite."""

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def create_image(path: Path, size: int = 32, color=(200, 80, 40)) -> Path:
    """Write a small real PNG image so decoding tests exercise actual codecs."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), color).save(path)
    return path


@pytest.fixture
def image_factory():
    """Expose :func:`create_image` to tests that need custom trees."""
    return create_image


@pytest.fixture
def synthetic_dataset(tmp_path):
    """A tiny Food-101-like layout: train/ and validation/ with class folders."""
    classes = ["apple_pie", "bibimbap", "pizza"]
    root = tmp_path / "food-101"
    train = root / "train"
    val = root / "validation"
    for ci, name in enumerate(classes):
        for i in range(6):
            create_image(train / name / f"{name}_{i}.png", color=(ci * 40, i * 10, 90))
        for i in range(2):
            create_image(val / name / f"{name}_v{i}.png", color=(ci * 40, i * 10, 10))
    return {
        "root": root,
        "train": train,
        "val": val,
        "classes": classes,
    }


@pytest.fixture
def synthetic_nutrition_csv(tmp_path):
    """A nutrition CSV with the real Food-101 column set and multi-weight rows."""
    csv_path = tmp_path / "inputs" / "nutrition.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = "label,weight,calories,protein,carbohydrates,fats,fiber,sugars,sodium"
    rows = [
        "apple_pie,100,237,2.4,34,11,1.6,15,266",
        "apple_pie,200,474,4.8,68,22,3.2,30,532",
        "bibimbap,100,130,5,18,4,1.2,2,300",
        "pizza,100,266,11,33,10,2.3,3.6,598",
    ]
    csv_path.write_text("\n".join([header] + rows), encoding="utf-8")
    return csv_path
