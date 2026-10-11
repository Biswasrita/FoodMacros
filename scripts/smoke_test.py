"""End-to-end smoke test on synthetic data (no external datasets required).

Trains a tiny classifier for one quick step, saves portable artifacts, reloads
them through the pipeline, and analyzes an image. This exercises the real code
paths (train -> save -> load -> classify -> weight -> nutrition -> report) so a
"the project runs" claim is backed by an actual execution rather than a guess.

Usage:
    python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from foodmacros.classification.train import train_classifier  # noqa: E402
from foodmacros.config import AppConfig  # noqa: E402
from foodmacros.data.splits import build_class_names, make_splits  # noqa: E402
from foodmacros.pipeline import FoodAnalysisPipeline  # noqa: E402
from foodmacros.reporting import format_analysis  # noqa: E402

CLASSES = ["apple_pie", "bibimbap", "pizza"]
NUTRITION_ROWS = [
    "apple_pie,100,237,2.4,34,11,1.6,15,266",
    "bibimbap,100,130,5,18,4,1.2,2,300",
    "pizza,100,266,11,33,10,2.3,3.6,598",
]


def _make_image(path: Path, size: int = 32, color=(200, 80, 40)) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), color).save(path)
    return path


def _write_synthetic_data(root: Path) -> tuple[Path, Path]:
    train = root / "train"
    val = root / "validation"
    for ci, name in enumerate(CLASSES):
        for i in range(3):
            _make_image(train / name / f"{name}_{i}.png", color=(ci * 60, i * 20, 90))
        _make_image(val / name / f"{name}_v.png", color=(ci * 60, 5, 10))
    return train, val


def _write_nutrition_csv(path: Path) -> Path:
    header = "label,weight,calories,protein,carbohydrates,fats,fiber,sugars,sodium"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([header] + NUTRITION_ROWS), encoding="utf-8")
    return path


def run_smoke(workdir: str | Path) -> dict:
    workdir = Path(workdir)
    train_root, val_root = _write_synthetic_data(workdir / "data")
    nutrition_csv = _write_nutrition_csv(workdir / "nutrition.csv")
    out_dir = workdir / "artifacts"

    config = AppConfig()
    config.model.img_size = 32
    config.model.num_classes = len(CLASSES)
    config.runtime.mode = "quick"
    config.training.quick_steps_per_epoch = 1
    config.training.batch_size = 2
    config.detection.enabled = False

    class_names = build_class_names(train_root, val_root, None)
    splits = make_splits(train_root, class_names, validation_root=val_root, val_fraction=0.0)
    train_result = train_classifier(
        config, class_names, splits.train, splits.val, output_dir=out_dir, weights=None, verbose=0
    )

    pipeline = FoodAnalysisPipeline.from_artifacts(
        config, model_dir=out_dir, nutrition_csv=nutrition_csv
    )
    sample = val_root / CLASSES[0] / f"{CLASSES[0]}_v.png"
    result = pipeline.analyze(str(sample), measured_grams=150, top_k=3)

    return {
        "model_path": train_result.model_path,
        "class_names": class_names,
        "report": format_analysis(result),
        "prediction_available": result.prediction.available,
        "weight_status": result.weight.status,
        "nutrition_basis": result.nutrition.basis,
    }


def main(argv=None) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        info = run_smoke(tmp)
        print(info["report"])
        print()
        print("Smoke test PASSED")
        print(f"  model           : {info['model_path']}")
        print(f"  prediction ok   : {info['prediction_available']}")
        print(f"  weight status   : {info['weight_status']}")
        print(f"  nutrition basis : {info['nutrition_basis']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
