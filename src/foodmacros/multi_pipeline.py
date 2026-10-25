"""Multi-dish analysis pipeline.

Flow for a single photo containing several foods:

  1. detect food boxes with the YOLO food detector
  2. crop each box from the **full-resolution** image (with a little padding)
  3. classify every crop with the Food-101 classifier
  4. look up macronutrients for each predicted dish
  5. report per-dish values and an honest plate total

Everything degrades cleanly. If detection is unavailable the whole image is
treated as a single dish (with a note); if a crop cannot be classified the dish
is reported as unavailable with a reason.

Weights are never invented. A weight is supplied **per dish** (``per_dish_grams``)
and is the only value treated as measured. Each dish's macros are shown for its
own weight when one is given, otherwise for a clearly-labelled reference
quantity. A **plate total is produced only when every *included* dish has both a
weight and nutrition data**; otherwise no total is shown and a reason is given.
A single number for the whole plate cannot be split between several dishes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .classification.model import get_preprocess_input
from .classification.predict import PredictionResult, load_classifier, predict_array
from .config import PROJECT_ROOT, AppConfig
from .data.preprocessing import ImageDecodeError
from .detection.crop import box_area, crop_array, load_full_image
from .detection.yolo_detector import Detection, DetectionResult, YoloDetector
from .nutrition.lookup import NutritionTable, VALID_STRATEGIES
from .nutrition.portion import NUTRIENT_COLUMNS, scale_to_grams
from .pipeline import NutritionOutcome
from .weight.auto_weight import MiDaSDepthProvider, build_depth_provider
from .weight.estimator import DepthEvidence, WeightEstimate, WeightEstimator

DEFAULT_CLASSIFIER_DIR = PROJECT_ROOT / "models" / "classifier"
DEFAULT_DETECTOR_WEIGHTS = PROJECT_ROOT / "models" / "food_detector" / "food_detector.pt"
DEFAULT_NUTRITION_CSV = PROJECT_ROOT / "data" / "nutrition.csv"


def _is_missing(value) -> bool:
    try:
        return value is None or math.isnan(float(value))
    except (TypeError, ValueError):
        return True


def _sum_nutrients(dicts: List[Dict[str, float]]) -> Dict[str, Optional[float]]:
    """Sum nutrient dicts; a nutrient is ``None`` if any dish is missing it."""
    keys: List[str] = []
    for d in dicts:
        for key in d:
            if key not in keys:
                keys.append(key)
    totals: Dict[str, Optional[float]] = {}
    for key in keys:
        values = [d.get(key) for d in dicts]
        if any(_is_missing(v) for v in values):
            totals[key] = None
        else:
            totals[key] = float(sum(float(v) for v in values))
    return totals


@dataclass
class DishResult:
    index: int
    detection: Optional[Detection]
    box: tuple = ()
    prediction: Optional[PredictionResult] = None
    weight: Optional[WeightEstimate] = None
    nutrition: Optional[NutritionOutcome] = None
    depth: Optional[DepthEvidence] = None
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def predicted_label(self) -> Optional[str]:
        if self.prediction is not None and self.prediction.available:
            return self.prediction.label
        return None

    def summary(self) -> str:
        det = self.detection
        det_text = f"{det.label} ({det.confidence:.0%})" if det else "whole image"
        pred = self.prediction
        if pred is None or not pred.available:
            reason = pred.reason if pred else "not run"
            return f"dish {self.index}: {det_text} -> classification unavailable ({reason})"
        return f"dish {self.index}: {det_text} -> {pred.label} ({pred.confidence:.1%})"


@dataclass
class MultiAnalysisResult:
    available: bool
    reason: str = ""
    detection: Optional[DetectionResult] = None
    dishes: List[DishResult] = field(default_factory=list)
    included: List[bool] = field(default_factory=list)  # per-dish: count in the total?
    totals: Dict[str, Optional[float]] = field(default_factory=dict)
    totals_basis: str = "none"  # "per_predicted_weight" | "none"
    totals_reason: str = ""
    total_grams: Optional[float] = None
    reference_quantity_g: float = 100.0
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class MultiDishPipeline:
    def __init__(
        self,
        config: AppConfig,
        classifier=None,
        class_names: Optional[List[str]] = None,
        detector: Optional[YoloDetector] = None,
        nutrition_table: Optional[NutritionTable] = None,
        weight_estimator: Optional[WeightEstimator] = None,
        depth_provider=None,
    ) -> None:
        self.config = config
        self.classifier = classifier
        self.class_names = list(class_names) if class_names else []
        self.detector = detector
        self.nutrition_table = nutrition_table
        self.weight_estimator = weight_estimator or WeightEstimator(config)
        self.depth_provider = depth_provider
        if self.depth_provider is None:
            try:
                self.depth_provider = build_depth_provider(config, self.weight_estimator)
            except Exception:
                self.depth_provider = None
        self.notes: List[str] = []
        if classifier is None or not self.class_names:
            self.notes.append("No classifier loaded: dish classification is unavailable.")
        if detector is None:
            self.notes.append(
                "No detector configured: the whole image is treated as a single dish."
            )
        if nutrition_table is None:
            self.notes.append("No nutrition table loaded: macronutrients are unavailable.")

    @property
    def strategy(self) -> str:
        strategy = self.config.nutrition.duplicate_resolution
        return strategy if strategy in VALID_STRATEGIES else "nearest_weight"

    # -- construction -----------------------------------------------------
    @classmethod
    def from_defaults(
        cls,
        config: AppConfig,
        model_dir=None,
        detector_weights=None,
        nutrition_csv=None,
        verify_class_names: bool = False,
    ) -> "MultiDishPipeline":
        """Build a pipeline from the project's default artifact locations.

        ``verify_class_names`` defaults to ``False`` because the food detector's
        own class list differs from Food-101; the detector is trusted to find
        food boxes and the classifier assigns the Food-101 label.
        """
        model_dir = Path(model_dir) if model_dir else DEFAULT_CLASSIFIER_DIR
        classifier = None
        class_names: List[str] = []
        if (model_dir / "classifier.keras").exists() and (model_dir / "class_names.json").exists():
            classifier, class_names = load_classifier(model_dir)

        detector = None
        if config.detection.enabled:
            weights = Path(detector_weights) if detector_weights else DEFAULT_DETECTOR_WEIGHTS
            detector = YoloDetector(
                weights=str(weights),
                expected_classes=class_names,
                conf_threshold=config.detection.conf_threshold,
                iou_threshold=config.detection.iou_threshold,
                verify_class_names=verify_class_names,
            )

        nutrition_table = None
        csv_path = Path(nutrition_csv) if nutrition_csv else DEFAULT_NUTRITION_CSV
        if csv_path.exists():
            try:
                nutrition_table = NutritionTable.from_csv(csv_path)
            except Exception:
                nutrition_table = None

        return cls(
            config=config,
            classifier=classifier,
            class_names=class_names,
            detector=detector,
            nutrition_table=nutrition_table,
        )

    # -- helpers ----------------------------------------------------------
    def _to_array(self, image) -> np.ndarray:
        if isinstance(image, (str, Path)):
            return load_full_image(image)
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"expected an HxWx3 RGB image, got shape {arr.shape}")
        return arr

    def _classify_crop(self, crop: np.ndarray, top_k: int) -> PredictionResult:
        if self.classifier is None or not self.class_names:
            return PredictionResult(False, reason="classifier not loaded")
        preprocess_fn = get_preprocess_input(self.config.model.architecture)
        return predict_array(
            self.classifier,
            preprocess_fn,
            crop,
            self.class_names,
            self.config.model.img_size,
            top_k=top_k,
        )

    def _nutrition(self, food_label: Optional[str], weight: WeightEstimate) -> NutritionOutcome:
        if self.nutrition_table is None:
            return NutritionOutcome(False, reason="nutrition data not loaded")
        if not food_label:
            return NutritionOutcome(False, reason="no predicted label to look up")
        if not self.nutrition_table.has_label(food_label):
            return NutritionOutcome(
                False, label=food_label, reason=f"'{food_label}' not in nutrition table"
            )
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

    def _set_dish_weight(self, dish: DishResult, grams: Optional[float]) -> None:
        """(Re)compute a dish's weight + nutrition for ``grams`` (or None)."""
        if grams is not None and str(grams).strip() != "":
            weight = self.weight_estimator.estimate(measured_grams=grams)
        else:
            weight = self.weight_estimator.estimate(depth=dish.depth, food_label=dish.predicted_label)
        dish.weight = weight
        dish.nutrition = self._nutrition(dish.predicted_label, weight)

    def _analyze_dish(
        self,
        index: int,
        detection: Optional[Detection],
        box,
        array: np.ndarray,
        top_k: int,
        crop_padding: float,
    ) -> DishResult:
        dish = DishResult(index=index, detection=detection, box=tuple(box))
        try:
            if detection is not None:
                crop = crop_array(array, box, padding=crop_padding)
            else:
                crop = array
        except ValueError as exc:
            dish.prediction = PredictionResult(False, reason=f"crop failed: {exc}")
            dish.nutrition = NutritionOutcome(False, reason="no crop to analyze")
            dish.weight = WeightEstimate("unavailable", None, "none")
            return dish

        dish.prediction = self._classify_crop(crop, top_k)

        depth = None
        if (
            self.config.weight.allow_experimental_depth
            and self.depth_provider is not None
            and detection is not None
        ):
            try:
                mask = np.zeros(crop.shape[:2], dtype=bool)
                x1, y1, x2, y2 = [int(round(float(v))) for v in box]
                px = (x2 - x1) * float(crop_padding)
                py = (y2 - y1) * float(crop_padding)
                ix1 = max(0, int(np.floor(x1 - px)))
                iy1 = max(0, int(np.floor(y1 - py)))
                ix2 = min(int(crop.shape[1]), int(np.ceil(x2 + px)))
                iy2 = min(int(crop.shape[0]), int(np.ceil(y2 + py)))
                if ix2 > ix1 and iy2 > iy1:
                    mask[iy1:iy2, ix1:ix2] = True
                depth = self.depth_provider(crop, dish.predicted_label, food_mask=mask)
            except Exception as exc:
                dish.warnings.append(f"depth estimation failed: {exc}")
        elif detection is None and self.config.weight.allow_experimental_depth and self.depth_provider is not None:
            try:
                depth = self.depth_provider(array, dish.predicted_label)
            except Exception as exc:
                dish.warnings.append(f"depth estimation failed: {exc}")

        dish.depth = depth
        self._set_dish_weight(dish, None)
        return dish

    def _set_weight_and_totals(
        self,
        result: MultiAnalysisResult,
        per_dish_grams: Optional[Sequence[Optional[float]]],
        include: Optional[Sequence[bool]],
    ) -> MultiAnalysisResult:
        """Apply per-dish weights to a result and recompute the plate total."""
        dishes = list(result.dishes)
        warnings = list(result.warnings)

        if per_dish_grams is not None:
            if len(per_dish_grams) != len(dishes):
                warnings.append(
                    f"got {len(per_dish_grams)} weights for {len(dishes)} dishes; "
                    "extra weights were ignored and missing ones left unset."
                )
            for dish, grams in zip(dishes, per_dish_grams):
                if grams is not None and str(grams).strip() != "":
                    self._set_dish_weight(dish, grams)

        included = self._resolve_include(include, len(dishes))
        totals, basis, total_grams, reason = self._totals(dishes, included)

        for dish in dishes:
            for message in (dish.warnings + (dish.weight.warnings if dish.weight else [])):
                if message not in warnings:
                    warnings.append(message)

        return replace(
            result,
            dishes=dishes,
            included=included,
            totals=totals,
            totals_basis=basis,
            totals_reason=reason,
            total_grams=total_grams,
            warnings=warnings,
        )

    @staticmethod
    def _resolve_include(include: Optional[Sequence[bool]], count: int) -> List[bool]:
        if include is None:
            return [True] * count
        flags = [bool(v) for v in include]
        if len(flags) < count:
            flags.extend([True] * (count - len(flags)))
        return flags[:count]

    # -- public API -------------------------------------------------------
    def analyze(
        self,
        image,
        top_k: int = 3,
        measured_grams: Optional[float] = None,
        per_dish_grams: Optional[Sequence[Optional[float]]] = None,
        include: Optional[Sequence[bool]] = None,
        max_dishes: Optional[int] = None,
        crop_padding: Optional[float] = None,
    ) -> MultiAnalysisResult:
        notes: List[str] = list(self.notes)
        warnings: List[str] = []

        try:
            array = self._to_array(image)
        except (ImageDecodeError, ValueError) as exc:
            return MultiAnalysisResult(available=False, reason=str(exc))

        max_dishes = int(max_dishes if max_dishes is not None else self.config.detection.max_dishes)
        crop_padding = float(
            crop_padding if crop_padding is not None else self.config.detection.crop_padding
        )

        detection = self.detector.detect(array) if self.detector is not None else None

        selected: List[Detection] = []
        if detection is not None and detection.available:
            ordered = sorted(detection.detections, key=lambda d: box_area(d.box_xyxy), reverse=True)
            selected = ordered[: max(max_dishes, 1)]
            if len(ordered) > len(selected):
                warnings.append(
                    f"{len(ordered)} dishes detected; only the {len(selected)} largest "
                    f"(max_dishes={max_dishes}) were analyzed."
                )
            if not selected:
                notes.append("No food detected; treating the whole image as one dish.")
        else:
            reason = detection.reason if detection is not None else "no detector configured"
            notes.append(f"Detection unavailable ({reason}); treating the whole image as one dish.")

        dishes: List[DishResult] = []
        if selected:
            for i, det in enumerate(selected, start=1):
                dishes.append(
                    self._analyze_dish(i, det, det.box_xyxy, array, top_k, crop_padding)
                )
        else:
            dishes.append(self._analyze_dish(1, None, (), array, top_k, crop_padding))

        # A single measured value can only describe a single dish.
        if per_dish_grams is None and measured_grams is not None:
            if len(dishes) == 1:
                per_dish_grams = [measured_grams]
            else:
                warnings.append(
                    f"measured weight ({measured_grams} g) ignored: it cannot be split "
                    f"across {len(dishes)} detected dishes; provide per-dish weights."
                )

        result = MultiAnalysisResult(
            available=True,
            detection=detection,
            dishes=dishes,
            included=[True] * len(dishes),
            reference_quantity_g=float(self.config.nutrition.reference_quantity_g),
            notes=notes,
            warnings=warnings,
        )
        return self._set_weight_and_totals(result, per_dish_grams, include)

    def apply_weights(
        self,
        result: MultiAnalysisResult,
        per_dish_grams: Optional[Sequence[Optional[float]]],
        include: Optional[Sequence[bool]] = None,
    ) -> MultiAnalysisResult:
        """Recompute a prior result with per-dish weights (no re-detection).

        Used by the UI: detect + classify once, then let the user confirm grams
        for each dish and recompute per-dish nutrition and the plate total.
        """
        if not result.available:
            return result
        return self._set_weight_and_totals(result, per_dish_grams, include)


    def _totals(
        self, dishes: List[DishResult], included: Sequence[bool]
    ) -> tuple[Dict[str, Optional[float]], str, Optional[float], str]:
        """Sum only the *included* dishes, and only when every one is complete.

        Returns ``(totals, basis, total_grams, reason)``. A plate total is
        produced only when every included dish has both an available weight and
        available nutrition; otherwise totals are empty and ``reason`` explains
        exactly what is missing.
        """
        included_dishes = [d for d, inc in zip(dishes, included) if inc]
        if not included_dishes:
            return {}, "none", None, "no plate total: no dishes are included."

        missing_weight = [
            d.index for d in included_dishes if d.weight is None or not d.weight.available
        ]
        missing_nutrition = [
            d.index for d in included_dishes if d.nutrition is None or not d.nutrition.available
        ]
        if missing_weight or missing_nutrition:
            parts = []
            if missing_weight:
                parts.append("missing weight for dish " + ", ".join(map(str, missing_weight)))
            if missing_nutrition:
                parts.append("missing nutrition for dish " + ", ".join(map(str, missing_nutrition)))
            return {}, "none", None, "no plate total: " + "; ".join(parts) + "."

        totals = _sum_nutrients([d.nutrition.nutrients for d in included_dishes])
        total_grams = float(sum(float(d.nutrition.grams) for d in included_dishes))
        return totals, "per_predicted_weight", total_grams, ""
