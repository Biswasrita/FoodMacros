"""Detection layer: guarded YOLO food detection."""

from .crop import box_area, clamp_box, crop_array, load_full_image
from .visualize import annotated_array, draw_detections
from .yolo_detector import Detection, DetectionResult, YoloDetector, is_available

__all__ = [
    "Detection",
    "DetectionResult",
    "YoloDetector",
    "is_available",
    "box_area",
    "clamp_box",
    "crop_array",
    "load_full_image",
    "annotated_array",
    "draw_detections",
]
