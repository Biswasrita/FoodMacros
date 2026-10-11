"""Stage 8 tests for the command-line scripts."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"script_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_scripts_import_and_expose_main():
    for name in ("inspect_inputs", "train_classifier", "run_analysis", "run_multi_analysis", "smoke_test"):
        module = _load(name)
        assert callable(module.main), name


def test_run_analysis_rejects_missing_image(tmp_path):
    module = _load("run_analysis")
    assert module.main([str(tmp_path / "nope.jpg")]) == 1


def test_run_multi_analysis_rejects_missing_image(tmp_path):
    module = _load("run_multi_analysis")
    assert module.main([str(tmp_path / "nope.jpg")]) == 1


def test_run_analysis_json_on_synthetic_image(image_factory, tmp_path, capsys):
    module = _load("run_analysis")
    image = image_factory(tmp_path / "meal.png")

    code = module.main([str(image), "--json", "--measured", "150"])
    assert code == 0

    payload = json.loads(capsys.readouterr().out.strip())
    assert "classification" in payload
    assert payload["weight"]["status"] == "measured"
    assert payload["weight"]["grams"] == 150


def test_run_analysis_text_report(image_factory, tmp_path, capsys):
    module = _load("run_analysis")
    image = image_factory(tmp_path / "meal.png")

    assert module.main([str(image)]) == 0
    out = capsys.readouterr().out
    assert "=== Food Analysis ===" in out
    assert "image_root" in out  # resolved.summary() was printed
