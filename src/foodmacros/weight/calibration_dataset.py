"""Calibration dataset management for real MiDaS-based weight estimation.

No fake measurements are created here. All entries must be provided by the user
with actual measured weights and real reference dimensions.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import PROJECT_ROOT

DEFAULT_CALIBRATION_DIR = PROJECT_ROOT / "data" / "calibration"
DEFAULT_CALIBRATION_SAMPLES = DEFAULT_CALIBRATION_DIR / "calibration_samples.json"
DEFAULT_CALIBRATION_IMAGES_DIR = DEFAULT_CALIBRATION_DIR / "images"
DEFAULT_CALIBRATED_PARAMS = PROJECT_ROOT / "models" / "calibration" / "depth_calibration.json"
DEFAULT_CALIBRATION_REPORT = PROJECT_ROOT / "models" / "calibration" / "calibration_report.json"


@dataclass
class CalibrationReference:
    type: str  # "known_size" | "known_height"
    label: str = ""
    known_cm: Dict[str, float] = field(default_factory=dict)
    pixels: Dict[str, float] = field(default_factory=dict)
    notes: str = ""


@dataclass
class CalibrationSample:
    id: str
    image_path: str
    food_label: Optional[str] = None
    measured_weight_g: Optional[float] = None
    references: List[CalibrationReference] = field(default_factory=list)
    roi: Dict[str, Any] = field(default_factory=dict)
    depth_stats: Dict[str, Any] = field(default_factory=dict)
    split: str = "train"  # train|val|holdout
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "image_path": self.image_path,
            "food_label": self.food_label,
            "measured_weight_g": self.measured_weight_g,
            "references": [
                {"type": r.type, "label": r.label, "known_cm": r.known_cm, "pixels": r.pixels, "notes": r.notes}
                for r in self.references
            ],
            "roi": self.roi,
            "depth_stats": self.depth_stats,
            "split": self.split,
            "notes": self.notes,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalibrationSample":
        refs = []
        for r in data.get("references") or []:
            refs.append(
                CalibrationReference(
                    type=str(r.get("type", "")),
                    label=str(r.get("label", "")),
                    known_cm=dict(r.get("known_cm") or {}),
                    pixels=dict(r.get("pixels") or {}),
                    notes=str(r.get("notes", "")),
                )
            )
        return cls(
            id=str(data.get("id", "")),
            image_path=str(data.get("image_path", "")),
            food_label=data.get("food_label"),
            measured_weight_g=data.get("measured_weight_g"),
            references=refs,
            roi=dict(data.get("roi") or {}),
            depth_stats=dict(data.get("depth_stats") or {}),
            split=str(data.get("split", "train")),
            notes=str(data.get("notes", "")),
            metadata=dict(data.get("metadata") or {}),
        )


def load_calibration_samples(path: str | Path | None = None) -> List[CalibrationSample]:
    p = Path(path) if path is not None else DEFAULT_CALIBRATION_SAMPLES
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"Failed to load calibration samples from {p}: {e}") from e
    samples_raw = data.get("samples") if isinstance(data, dict) else data
    if not isinstance(samples_raw, list):
        raise ValueError("Calibration file must contain a list of samples or {'samples': [...]}")
    samples = []
    for s in samples_raw:
        if isinstance(s, dict):
            samples.append(CalibrationSample.from_dict(s))
    return samples


def save_calibration_samples(samples: List[CalibrationSample], path: str | Path | None = None) -> Path:
    p = Path(path) if path is not None else DEFAULT_CALIBRATION_SAMPLES
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "samples": [s.to_dict() for s in samples]}
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


def split_samples(samples: List[CalibrationSample], train_frac: float = 0.6, val_frac: float = 0.2, seed: int = 42) -> Dict[str, List[CalibrationSample]]:
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1.0")
    rng = random.Random(seed)
    items = list(samples)
    rng.shuffle(items)
    n = len(items)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return {
        "train": items[:n_train],
        "val": items[n_train : n_train + n_val],
        "holdout": items[n_train + n_val :],
    }


def validate_sample(sample: CalibrationSample) -> List[str]:
    errors = []
    if not sample.id:
        errors.append("missing id")
    if not sample.image_path:
        errors.append("missing image_path")
    if sample.measured_weight_g is None:
        errors.append("missing measured_weight_g (real kitchen scale measurement required)")
    else:
        try:
            w = float(sample.measured_weight_g)
            if w <= 0 or not (w == w):  # NaN check
                errors.append("measured_weight_g must be positive finite number")
        except Exception:
            errors.append("measured_weight_g must be a number")
    if not sample.references:
        errors.append("at least one reference object (known size/height) required")
    else:
        has_known_size = False
        for r in sample.references:
            if r.type == "known_size":
                has_known_size = True
                if not r.known_cm or not r.pixels:
                    errors.append("known_size reference requires known_cm and pixels")
        if not has_known_size:
            errors.append("need at least one known_size reference to compute cm_per_pixel")
    return errors
