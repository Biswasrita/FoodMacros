"""Stage 1 tests for environment detection helpers."""

from pathlib import Path

from foodmacros.env_checks import (
    default_working_dir,
    format_report,
    get_env_report,
    has_package,
)


def test_has_package_detects_known_and_unknown():
    assert has_package("numpy") is True
    assert has_package("a_package_that_does_not_exist_xyz") is False


def test_env_report_has_expected_structure():
    report = get_env_report(check_tensorflow=True)
    for key in (
        "python_version",
        "platform",
        "is_kaggle",
        "working_dir",
        "packages",
        "missing_required",
        "gpu_available",
    ):
        assert key in report
    assert isinstance(report["packages"], dict)
    assert isinstance(report["missing_required"], list)


def test_default_working_dir_is_a_path():
    assert isinstance(default_working_dir(), Path)


def test_format_report_includes_python_version():
    report = get_env_report(check_tensorflow=False)
    text = format_report(report)
    assert "Python" in text
    assert report["python_version"] in text
