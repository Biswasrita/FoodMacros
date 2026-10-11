"""Stage 6 tests for the guarded YOLO detector (ultralytics optional)."""

from pathlib import Path

import pytest

from foodmacros.detection.yolo_detector import (
    Detection,
    DetectionResult,
    YoloDetector,
    _normalize,
    is_available,
)
from foodmacros.env_checks import has_package


def test_is_available_matches_ultralytics_presence():
    assert is_available() is has_package("ultralytics")


def test_normalize_names_dict_and_list():
    assert _normalize({1: "b", 0: "a"}) == ["a", "b"]
    assert _normalize(["x", "y"]) == ["x", "y"]


def test_no_weights_is_unavailable():
    detector = YoloDetector(weights=None).load()
    assert detector.load_error
    result = detector.detect("whatever.jpg")
    assert not result.available
    assert result.reason


def test_missing_weights_file_reports_error():
    detector = YoloDetector(weights="definitely_missing.pt").load()
    assert detector.load_error
    # The exact message depends on whether ultralytics is installed, but it
    # must always be actionable and non-empty.
    assert "ultralytics" in detector.load_error.lower() or "not found" in detector.load_error.lower()


def test_class_check_rejects_non_food_model():
    detector = YoloDetector(expected_classes=["pizza", "bibimbap"], verify_class_names=True)
    error = detector._class_check_error(["person", "car", "dog"])
    assert error and "non-food" in error


def test_class_check_accepts_matching_food_model():
    detector = YoloDetector(expected_classes=["pizza", "bibimbap"], verify_class_names=True)
    assert detector._class_check_error(["pizza", "burger"]) is None


def test_class_check_is_case_insensitive():
    detector = YoloDetector(expected_classes=["Pizza"], verify_class_names=True)
    assert detector._class_check_error(["PIZZA"]) is None


def test_class_check_can_be_disabled():
    detector = YoloDetector(expected_classes=["pizza"], verify_class_names=False)
    assert detector._class_check_error(["person"]) is None


def test_detection_result_summary_and_labels():
    result = DetectionResult(
        True, detections=[Detection("pizza", 0.9, (0, 0, 10, 10)), Detection("salad", 0.5, (0, 0, 5, 5))]
    )
    assert result.labels() == ["pizza", "salad"]
    assert "pizza" in result.summary()

    empty = DetectionResult(True, detections=[])
    assert "no food detected" in empty.summary()

    unavailable = DetectionResult(False, reason="no model")
    assert "unavailable" in unavailable.summary()
