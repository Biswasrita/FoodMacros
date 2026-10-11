"""Guarded YOLO food-detection wrapper.

Ultralytics YOLO ships pretrained on COCO (80 non-food classes). This module
therefore *verifies* the loaded model's class names against an expected food
class set before it is used. If ``ultralytics`` is missing, no compatible
weights are provided, or the model's classes do not match, detection reports
"unavailable" with an actionable reason instead of returning bogus boxes.

Bounding boxes are detections only; they are NEVER treated as weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from ..env_checks import has_package


@dataclass
class Detection:
    label: str
    confidence: float
    box_xyxy: tuple  # (x1, y1, x2, y2) in pixels


@dataclass
class DetectionResult:
    available: bool
    detections: List[Detection] = field(default_factory=list)
    reason: str = ""

    def labels(self) -> List[str]:
        return [d.label for d in self.detections]

    def summary(self) -> str:
        if not self.available:
            return f"detection unavailable: {self.reason}"
        if not self.detections:
            return "detection available: no food detected"
        return "; ".join(f"{d.label} ({d.confidence:.0%})" for d in self.detections)


def is_available() -> bool:
    return has_package("ultralytics")


def _normalize(names) -> List[str]:
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names)]
    return [str(n) for n in names]


class YoloDetector:
    """Thin wrapper around an Ultralytics model with class verification.

    ``expected_classes`` should be the project's known food class set (e.g. the
    Food-101 ``class_names``). When provided and ``verify_class_names`` is set,
    a model whose classes do not overlap the expected set is rejected.
    """

    def __init__(
        self,
        weights: Optional[str] = None,
        expected_classes: Optional[Sequence[str]] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        verify_class_names: bool = True,
    ) -> None:
        self.weights = weights
        self.expected_classes = {str(c).strip().lower() for c in (expected_classes or [])}
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.verify_class_names = verify_class_names
        self._model = None
        self.class_names: List[str] = []
        self.load_error: str = ""

    # -- loading ----------------------------------------------------------
    def load(self) -> "YoloDetector":
        """Load the model once. Sets ``load_error`` and leaves it unusable on failure."""
        if self._model is not None or self.load_error:
            return self
        if not has_package("ultralytics"):
            self.load_error = (
                "ultralytics is not installed; install it (pip install ultralytics) "
                "to enable detection."
            )
            return self
        if not self.weights:
            self.load_error = (
                "no YOLO weights provided. COCO weights contain no food classes; "
                "supply a food-detection .pt (config.data.detection_weights)."
            )
            return self
        if not Path(self.weights).exists():
            self.load_error = f"YOLO weights file not found: {self.weights}"
            return self

        from ultralytics import YOLO

        try:
            self._model = YOLO(self.weights)
        except Exception as exc:  # surface model-load problems clearly
            self.load_error = f"failed to load YOLO weights '{self.weights}': {exc}"
            return self

        self.class_names = _normalize(getattr(self._model, "names", []) or [])
        error = self._class_check_error(self.class_names)
        if error:
            self.load_error = error
            self._model = None
        return self

    def _class_check_error(self, class_names: Sequence[str]) -> Optional[str]:
        """Return an error message if the model's classes are not food-like."""
        if not self.verify_class_names:
            return None
        overlap = self.expected_classes & {c.lower() for c in class_names}
        if not overlap:
            return (
                "loaded YOLO model has no classes in common with the expected food "
                f"classes (model has {len(class_names)} classes, e.g. "
                f"{list(class_names)[:5]}). This looks like a non-food model; "
                "provide food-detection weights."
            )
        return None

    # -- inference --------------------------------------------------------
    def detect(self, image) -> DetectionResult:
        """Run detection on a path or an HxWx3 array."""
        self.load()
        if self._model is None:
            return DetectionResult(False, reason=self.load_error or "model not loaded")
        try:
            results = self._model.predict(
                source=image, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False
            )
        except Exception as exc:
            return DetectionResult(False, reason=f"detection failed: {exc}")

        detections: List[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            names = getattr(result, "names", None) or {}
            for box in boxes:
                cls_id = int(box.cls[0]) if hasattr(box, "cls") else -1
                label = str(names.get(cls_id, cls_id)) if isinstance(names, dict) else str(cls_id)
                confidence = float(box.conf[0]) if hasattr(box, "conf") else float("nan")
                xyxy = tuple(float(v) for v in box.xyxy[0].tolist()) if hasattr(box, "xyxy") else ()
                detections.append(Detection(label=label, confidence=confidence, box_xyxy=xyxy))
        return DetectionResult(True, detections=detections)
