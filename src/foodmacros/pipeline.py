"""End-to-end analysis pipeline.

Ties together the independent modules (classification, detection, weight,
nutrition) behind a single ``analyze()`` call. Every stage degrades cleanly:
a missing model/source produces an "unavailable" result with a reason, and no
value is ever invented.

Stage map:
  1. discover inputs (via ``from_discovery``)
  2. validate the image
  3. classify the dish (top-k)
  4. optional detection
  5. estimate portion weight (measured / regression / experimental)
  6. compute macros for the (predicted or reference) weight
  7. assemble a structured result
  8. report (see ``reporting.py``)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .classification.model import get_preprocess_input
from .classification.predict import (
    PredictionResult,
    load_classifier,
    predict_array,
    predict_path,
)
from .config import AppConfig, PROJECT_ROOT
from .data.discovery import ResolvedInputs, resolve_inputs
from .data.preprocessing import ImageDecodeError, load_image
from .data.splits import build_class_names
from .detection.yolo_detector import DetectionResult, YoloDetector
from .nutrition.lookup import NutritionTable, VALID_STRATEGIES
from .nutrition.portion import scale_to_grams
from .weight.estimator import WeightEstimate, WeightEstimator

# Default runtime artifacts bundled with the project (project-relative). These
# are copies of the trained originals (see docs); originals are never renamed,
# deleted, or overwritten.
DEFAULT_CLASSIFIER_DIR: Path = PROJECT_ROOT / "models" / "classifier"
DEFAULT_DETECTOR_WEIGHTS: Path = (
    PROJECT_ROOT / "models" / "food_detector" / "food_detector.pt"
)


def _is_bundled_detector(path: str | Path | None) -> bool:
    """True when ``path`` is the bundled food detector."""
    if not path:
        return False
    try:
        return Path(path).resolve() == DEFAULT_DETECTOR_WEIGHTS.resolve()
    except (OSError, ValueError):
        return False


def _resolve_default_artifacts(
    config: AppConfig, resolved: ResolvedInputs, model_dir=None
):
    """Resolve the classifier directory and detector weights to use.

    Explicit values always win. Otherwise fall back to the bundled project
    artifacts so the single-dish pipeline works out of the box, mirroring the
    multi-dish pipeline's defaults. Returns ``(model_dir, detection_weights,
    notes)`` where ``notes`` drops the stale "no detector" note when the bundled
    detector is used.
    """
    if model_dir:
        classifier_dir = Path(model_dir)
    elif (
        (DEFAULT_CLASSIFIER_DIR / "classifier.keras").exists()
        and (DEFAULT_CLASSIFIER_DIR / "class_names.json").exists()
    ):
        classifier_dir = DEFAULT_CLASSIFIER_DIR
    else:
        classifier_dir = None

    detection_weights = resolved.detection_weights
    notes = list(resolved.notes)
    if config.detection.enabled and not detection_weights and DEFAULT_DETECTOR_WEIGHTS.exists():
        detection_weights = str(DEFAULT_DETECTOR_WEIGHTS)
        notes = [n for n in notes if "detection weight" not in n.lower()]

    return classifier_dir, detection_weights, notes


@dataclass
class NutritionOutcome:
    available: bool
    label: Optional[str] = None
    basis: str = "none"  # "per_predicted_weight" | "per_reference" | "none"
    grams: Optional[float] = None
    nutrients: Dict[str, float] = field(default_factory=dict)
    per_100g: Dict[str, float] = field(default_factory=dict)
    note: str = ""
    reason: str = ""


@dataclass
class AnalysisResult:
    prediction: Optional[PredictionResult]
    detection: Optional[DetectionResult]
    weight: WeightEstimate
    nutrition: NutritionOutcome
    reference_quantity_g: float
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def _load_regression(model_dir: str | Path, config: AppConfig) -> Optional[Dict[str, Any]]:
    """Load a saved mass regressor + normalisation stats, if both exist."""
    from .weight.regression import get_regression_preprocess_input

    model_dir = Path(model_dir)
    model_path = model_dir / "mass_regressor.keras"
    stats_path = model_dir / "regression_norm_stats.json"
    if not model_path.exists() or not stats_path.exists():
        return None
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path, compile=False)
    raw = json.loads(stats_path.read_text(encoding="utf-8"))
    stats = {k: (float(v[0]), float(v[1])) for k, v in raw["stats"].items()}
    return {
        "model": model,
        "stats": stats,
        "img_size": config.model.img_size,
        "preprocess_fn": get_regression_preprocess_input(config.weight.regression_backbone),
    }


class FoodAnalysisPipeline:
    def __init__(
        self,
        config: AppConfig,
        classifier=None,
        class_names: Optional[List[str]] = None,
        nutrition_table: Optional[NutritionTable] = None,
        detector: Optional[YoloDetector] = None,
        weight_estimator: Optional[WeightEstimator] = None,
        regression: Optional[Dict[str, Any]] = None,
        depth_provider=None,
    ) -> None:
        self.config = config
        self.classifier = classifier
        self.class_names = list(class_names) if class_names else []
        self.nutrition_table = nutrition_table
        self.detector = detector
        self.weight_estimator = weight_estimator or WeightEstimator(config)
        self.regression = regression
        self.depth_provider = depth_provider

        self.notes: List[str] = []
        if classifier is None or not self.class_names:
            self.notes.append("No classifier loaded: classification is unavailable.")
        if nutrition_table is None:
            self.notes.append("No nutrition table loaded: macronutrients are unavailable.")
        if regression is None:
            self.notes.append(
                "No mass regressor loaded: portion weight reports 'unavailable' "
                "unless a measured weight is supplied."
            )

    @property
    def strategy(self) -> str:
        strategy = self.config.nutrition.duplicate_resolution
        return strategy if strategy in VALID_STRATEGIES else "nearest_weight"

    # -- construction -----------------------------------------------------
    @classmethod
    def from_artifacts(
        cls,
        config: AppConfig,
        model_dir=None,
        nutrition_csv=None,
        detection_weights=None,
        mass_regressor_dir=None,
        expected_classes=None,
    ) -> "FoodAnalysisPipeline":
        classifier = None
        class_names: List[str] = []

        if model_dir is not None:
            model_dir = Path(model_dir)
            if (model_dir / "classifier.keras").exists() and (model_dir / "class_names.json").exists():
                classifier, class_names = load_classifier(model_dir)

        if not class_names and expected_classes:
            class_names = [str(c) for c in expected_classes]

        nutrition_table = None
        if nutrition_csv:
            try:
                nutrition_table = NutritionTable.from_csv(nutrition_csv)
            except Exception:
                nutrition_table = None

        detector = None
        if config.detection.enabled and detection_weights:
            # The bundled detector is a 113-class food model whose class list
            # differs from Food-101 (same situation as the multi-dish path), so
            # it is loaded without name verification. Custom detectors keep the
            # configured behaviour.
            verify = config.detection.verify_class_names and not _is_bundled_detector(
                detection_weights
            )
            detector = YoloDetector(
                weights=str(detection_weights),
                expected_classes=class_names,
                conf_threshold=config.detection.conf_threshold,
                iou_threshold=config.detection.iou_threshold,
                verify_class_names=verify,
            )

        regression = None
        if config.weight.enable_regression and mass_regressor_dir is not None:
            regression = _load_regression(mass_regressor_dir, config)

        return cls(
            config=config,
            classifier=classifier,
            class_names=class_names,
            nutrition_table=nutrition_table,
            detector=detector,
            regression=regression,
        )

    @classmethod
    def from_discovery(
        cls, config: AppConfig, model_dir=None, mass_regressor_dir=None
    ) -> tuple["FoodAnalysisPipeline", Any]:
        resolved = resolve_inputs(config)
        class_names = None
        if resolved.image_root:
            try:
                class_names = build_class_names(
                    resolved.image_root, resolved.validation_root, config.data.class_names_file
                )
            except Exception:
                class_names = None

        model_dir, detection_weights, notes = _resolve_default_artifacts(
            config, resolved, model_dir=model_dir
        )
        resolved.detection_weights = detection_weights
        resolved.notes = notes
        pipeline = cls.from_artifacts(
            config,
            model_dir=model_dir,
            nutrition_csv=resolved.nutrition_csv,
            detection_weights=detection_weights,
            mass_regressor_dir=mass_regressor_dir,
            expected_classes=class_names,
        )
        pipeline.notes.extend(notes)
        pipeline.notes.extend(f"MISSING: {m}" for m in resolved.missing)
        return pipeline, resolved

    # -- stages -----------------------------------------------------------
    def _as_array(self, image) -> Optional[np.ndarray]:
        if image is None:
            return None
        if isinstance(image, (str, Path)):
            try:
                return load_image(image, self.config.model.img_size)
            except ImageDecodeError:
                return None
        array = np.asarray(image)
        return array if array.size else None

    def _classify(self, image, top_k: int) -> PredictionResult:
        if self.classifier is None or not self.class_names:
            return PredictionResult(False, reason="classifier not loaded")
        preprocess_fn = get_preprocess_input(self.config.model.architecture)
        img_size = self.config.model.img_size
        if isinstance(image, (str, Path)):
            return predict_path(self.classifier, preprocess_fn, image, self.class_names, img_size, top_k=top_k)
        return predict_array(self.classifier, preprocess_fn, image, self.class_names, img_size, top_k=top_k)

    def _detect(self, image) -> DetectionResult:
        if not self.config.detection.enabled:
            return DetectionResult(False, reason="detection disabled in config")
        if self.detector is None:
            return DetectionResult(False, reason="no detector configured")
        return self.detector.detect(str(image) if isinstance(image, (str, Path)) else image)

    def _regression_prediction(self, array):
        if self.regression is None or array is None:
            return None
        from .weight.regression import predict_mass

        try:
            return predict_mass(
                self.regression["model"],
                self.regression["preprocess_fn"],
                array,
                self.regression["img_size"],
                self.regression["stats"],
            )
        except Exception:
            return None

    def _depth_evidence(self, array, food_label, warnings: List[str]):
        if not self.config.weight.allow_experimental_depth or self.depth_provider is None:
            return None
        try:
            return self.depth_provider(array, food_label)
        except Exception as exc:
            warnings.append(f"depth provider failed: {exc}")
            return None

    def _nutrition(self, food_label, weight: WeightEstimate) -> NutritionOutcome:
        if self.nutrition_table is None:
            return NutritionOutcome(False, reason="nutrition data not loaded")
        if not food_label:
            return NutritionOutcome(False, reason="no predicted label to look up")
        if not self.nutrition_table.has_label(food_label):
            return NutritionOutcome(False, label=food_label, reason=f"'{food_label}' not in nutrition table")

        per_100g = self.nutrition_table.per_100g(food_label, strategy=self.strategy)
        if weight.available and weight.grams:
            grams = float(weight.grams)
            basis = "per_predicted_weight"
            note = f"macros for the {weight.status} weight ({grams:g} g)"
        else:
            grams = float(self.config.nutrition.reference_quantity_g)
            basis = "per_reference"
            note = (
                f"weight unavailable, so macros are shown for the reference "
                f"quantity ({grams:g} g) only"
            )
        nutrients = scale_to_grams(per_100g, grams)
        return NutritionOutcome(
            available=True,
            label=food_label,
            basis=basis,
            grams=grams,
            nutrients=nutrients,
            per_100g=per_100g,
            note=note,
        )

    def analyze(self, image, measured_grams: Optional[float] = None, top_k: int = 3) -> AnalysisResult:
        warnings: List[str] = []

        array = self._as_array(image)
        if array is None and not isinstance(image, (str, Path)):
            warnings.append("invalid or empty image")

        prediction = self._classify(image, top_k=top_k)
        detection = self._detect(image)

        food_label = prediction.label if prediction.available else None
        regression_pred = self._regression_prediction(array)
        depth = self._depth_evidence(array, food_label, warnings)

        weight = self.weight_estimator.estimate(
            measured_grams=measured_grams,
            regression=regression_pred,
            depth=depth,
            food_label=food_label,
        )
        warnings.extend(weight.warnings)

        nutrition = self._nutrition(food_label, weight)

        return AnalysisResult(
            prediction=prediction,
            detection=detection,
            weight=weight,
            nutrition=nutrition,
            reference_quantity_g=float(self.config.nutrition.reference_quantity_g),
            notes=list(self.notes),
            warnings=warnings,
        )
