"""Central configuration for the FoodMacros project.

All tunable settings live here and/or in a JSON file loaded at runtime.
No machine-specific absolute paths are hard-coded: input locations are
resolved at runtime (e.g. under ``/kaggle/input``) and may be overridden
through the ``data`` section. Output locations default to a runtime-resolved
directory, not a developer machine path.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

# Project root = .../Final_Year_Project (src/foodmacros/config.py -> parents[2])
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "configs" / "default.json"


@dataclass
class RuntimeConfig:
    """Runtime behaviour that is independent of any specific machine."""

    seed: int = 42
    mode: str = "quick"  # "quick" | "full"
    mixed_precision: str = "auto"  # "auto" | "float32" | "mixed_float16"
    output_dir: str | None = None  # None -> resolved at runtime by env_checks


@dataclass
class DataConfig:
    """Dataset locations. ``None`` means "discover at runtime".

    These are intentionally left unset so that the pipeline resolves the real
    location of attached Kaggle datasets instead of assuming a fixed path.
    """

    image_root: str | None = None
    validation_root: str | None = None
    nutrition_csv: str | None = None
    nutrition5k_root: str | None = None
    detection_weights: str | None = None
    class_names_file: str | None = None
    max_classes: int | None = None
    validate_images: bool = True


@dataclass
class SplitConfig:
    val_fraction: float = 0.10
    test_source: str = "validation"  # "validation" | "carve_from_train"
    shuffle_seed: int = 42


@dataclass
class ModelConfig:
    architecture: str = "efficientnetb0"
    use_vit_fusion: bool = False
    vit_preset: str = "vit_base_patch16_224"
    img_size: int = 224
    dropout: float = 0.3
    num_classes: int = 101


@dataclass
class TrainingConfig:
    batch_size: int = 32
    epochs_head: int = 5
    epochs_finetune: int = 10
    quick_epochs: int = 1
    quick_steps_per_epoch: int = 20
    learning_rate_head: float = 1e-3
    learning_rate_finetune: float = 1e-5
    label_smoothing: float = 0.1
    early_stopping_patience: int = 5


@dataclass
class NutritionConfig:
    reference_quantity_g: float = 100.0
    derive_per_100g: bool = True
    duplicate_resolution: str = "nearest_weight"  # documented rule for multi-row lookup
    require_class_match: bool = True


@dataclass
class WeightConfig:
    enable_regression: bool = True
    regression_backbone: str = "mobilenetv2"
    allow_experimental_depth: bool = False
    min_reliable_grams: float = 5.0
    max_reliable_grams: float = 2000.0
    # Physical calibration for the experimental depth path. These are never
    # assumed: when they are missing the depth path reports "unavailable".
    # ``cm_per_pixel`` is the image-plane scale from a known-size reference and
    # ``cm_per_depth_unit`` maps relative depth units to centimetres. When
    # ``reference_depth`` is omitted the minimum depth in the crop is used as
    # the reference plane and a warning is reported.
    cm_per_pixel: float | None = None
    cm_per_depth_unit: float | None = None
    reference_depth: float | None = None
    invert_depth: bool = False

    def __post_init__(self) -> None:
        for name in ("cm_per_pixel", "cm_per_depth_unit"):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"weight.{name} must be a positive, finite number when provided.")
        if self.reference_depth is not None:
            if (
                not isinstance(self.reference_depth, (int, float))
                or not math.isfinite(float(self.reference_depth))
            ):
                raise ValueError("weight.reference_depth must be a finite number when provided.")


@dataclass
class DetectionConfig:
    enabled: bool = True
    model: str = "yolo11n.pt"
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    verify_class_names: bool = True
    # Multi-dish behaviour (used by the multi-dish pipeline only).
    max_dishes: int = 5
    crop_padding: float = 0.08


@dataclass
class UIConfig:
    share: bool = False
    server_port: int | None = None
    max_top_k: int = 3


@dataclass
class AppConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    nutrition: NutritionConfig = field(default_factory=NutritionConfig)
    weight: WeightConfig = field(default_factory=WeightConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    ui: UIConfig = field(default_factory=UIConfig)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


_SECTIONS = {
    "runtime": RuntimeConfig,
    "data": DataConfig,
    "splits": SplitConfig,
    "model": ModelConfig,
    "training": TrainingConfig,
    "nutrition": NutritionConfig,
    "weight": WeightConfig,
    "detection": DetectionConfig,
    "ui": UIConfig,
}


def _build_section(cls: type, data: Dict[str, Any]) -> Any:
    valid = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - valid
    if unknown:
        raise ValueError(
            f"Unknown key(s) {sorted(unknown)} in config section '{cls.__name__}'."
        )
    return cls(**data)


def config_from_dict(data: Dict[str, Any]) -> AppConfig:
    """Build an :class:`AppConfig` from a nested mapping.

    Missing sections fall back to their dataclass defaults; unknown keys raise
    a clear error so typos are caught instead of silently ignored.
    """
    if not isinstance(data, dict):
        raise ValueError("Config root must be a JSON object.")
    unknown = set(data) - set(_SECTIONS)
    if unknown:
        raise ValueError(f"Unknown config section(s): {sorted(unknown)}")
    sections = {
        name: _build_section(cls, data.get(name, {}))
        for name, cls in _SECTIONS.items()
    }
    return AppConfig(**sections)


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load configuration from ``path`` (or the default JSON) with fallbacks."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if config_path.exists():
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        return config_from_dict(raw)
    return AppConfig()
