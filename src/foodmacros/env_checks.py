"""Environment detection and safe defaults.

This module inspects the runtime (Python, TensorFlow, GPU, optional packages)
and sets a mixed-precision policy accordingly. It never raises on missing
optional dependencies and never imports heavy packages unless needed.
"""

from __future__ import annotations

import importlib.util
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List

from .config import PROJECT_ROOT

# Optional dependencies the project can use but does not require for the
# core (classification + nutrition) functionality.
OPTIONAL_PACKAGES: List[str] = [
    "numpy",
    "pandas",
    "sklearn",
    "PIL",
    "gradio",
    "ultralytics",
    "torch",
    "keras_hub",
    "pytest",
]

_REQUIRED_PACKAGES: List[str] = ["numpy"]


def has_package(name: str) -> bool:
    """Return True if ``name`` is importable, without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _package_version(name: str) -> str | None:
    try:
        import importlib.metadata as md

        return md.version(name.replace("_", "-"))
    except Exception:
        return None


def default_working_dir() -> Path:
    """Return a sensible output directory for the current environment.

    Uses ``/kaggle/working`` when running on Kaggle, otherwise a local
    ``outputs`` directory next to the project. No hard-coded user path.
    """
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.exists():
        return kaggle_working
    return PROJECT_ROOT / "outputs"


def get_env_report(check_tensorflow: bool = True) -> Dict[str, Any]:
    """Collect a structured, honest report about the current environment."""
    report: Dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "is_kaggle": Path("/kaggle/input").exists(),
        "working_dir": str(default_working_dir()),
        "packages": {name: has_package(name) for name in OPTIONAL_PACKAGES},
        "package_versions": {
            name: _package_version(name)
            for name in OPTIONAL_PACKAGES
            if has_package(name)
        },
        "missing_required": [p for p in _REQUIRED_PACKAGES if not has_package(p)],
        "tensorflow": None,
        "gpus": [],
    }

    if check_tensorflow and has_package("tensorflow"):
        try:
            import tensorflow as tf  # local import: heavy

            report["tensorflow"] = tf.__version__
            report["gpus"] = [d.name for d in tf.config.list_physical_devices("GPU")]
            report["gpu_available"] = bool(report["gpus"])
        except Exception as exc:  # pragma: no cover - environment dependent
            report["tensorflow"] = None
            report["tensorflow_error"] = f"{type(exc).__name__}: {exc}"
            report["gpu_available"] = False
    else:
        report["gpu_available"] = False

    return report


def configure_mixed_precision(
    policy: str = "auto", report: Dict[str, Any] | None = None
) -> str:
    """Apply a mixed-precision policy and return the one actually applied.

    ``auto`` selects ``mixed_float16`` only when a GPU is present, otherwise
    ``float32``. Explicit ``mixed_float16`` is only honoured when a GPU exists,
    because mixed precision on CPU is slower and unsupported in places.
    """
    if policy not in {"auto", "float32", "mixed_float16"}:
        raise ValueError(f"Unknown mixed_precision policy: {policy!r}")

    if report is None:
        report = get_env_report()

    if not report.get("tensorflow"):
        raise RuntimeError(
            "TensorFlow is not available; cannot configure mixed precision."
        )

    import tensorflow as tf

    gpu = bool(report.get("gpu_available"))
    resolved = "float32"
    if policy == "auto" and gpu:
        resolved = "mixed_float16"
    elif policy == "mixed_float16":
        resolved = "mixed_float16" if gpu else "float32"

    tf.keras.mixed_precision.set_global_policy(resolved)
    return resolved


def format_report(report: Dict[str, Any]) -> str:
    """Human-readable summary of an environment report."""
    lines = [
        f"Python          : {report['python_version']}",
        f"Platform        : {report['platform']}",
        f"Kaggle (/kaggle/input): {report['is_kaggle']}",
        f"Working dir     : {report['working_dir']}",
        f"TensorFlow      : {report.get('tensorflow') or 'not available'}",
        f"GPUs            : {report.get('gpus') or 'none'}",
    ]
    present = [k for k, v in report["packages"].items() if v]
    absent = [k for k, v in report["packages"].items() if not v]
    lines.append(f"Packages present: {', '.join(present) or 'none'}")
    lines.append(f"Packages absent : {', '.join(absent) or 'none'}")
    if report.get("missing_required"):
        lines.append(f"MISSING REQUIRED: {', '.join(report['missing_required'])}")
    return "\n".join(lines)
