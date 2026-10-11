"""Tests for bounding-box cropping utilities used by the multi-dish pipeline."""

import numpy as np
import pytest

from foodmacros.detection.crop import box_area, clamp_box, crop_array, load_full_image
from foodmacros.detection.visualize import annotated_array, draw_detections
from foodmacros.detection.yolo_detector import Detection


def test_clamp_box_without_padding():
    assert clamp_box((10, 20, 30, 40), width=100, height=100, padding=0.0) == (10, 20, 30, 40)


def test_clamp_box_pads_and_clamps_to_image():
    assert clamp_box((10, 10, 20, 20), width=25, height=25, padding=0.5) == (5, 5, 25, 25)


def test_clamp_box_orders_reversed_coordinates():
    assert clamp_box((30, 40, 10, 20), width=100, height=100) == (10, 20, 30, 40)


def test_crop_array_returns_expected_region():
    array = np.zeros((10, 10, 3), dtype=np.uint8)
    array[2:5, 3:7] = 255
    crop = crop_array(array, (3, 2, 7, 5), padding=0.0)
    assert crop.shape == (3, 4, 3)
    assert (crop == 255).all()


def test_crop_array_degenerate_box_raises():
    array = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="empty crop"):
        crop_array(array, (5, 5, 5, 5))


def test_box_area():
    assert box_area((0, 0, 4, 5)) == 20.0
    assert box_area((4, 4, 0, 0)) == 16.0
    assert box_area((1, 1, 1, 5)) == 0.0


def test_load_full_image_preserves_native_size(image_factory, tmp_path):
    from PIL import Image

    path = tmp_path / "wide.png"
    Image.new("RGB", (60, 40), (10, 20, 30)).save(path)
    array = load_full_image(path)
    assert array.shape == (40, 60, 3)
    assert array.dtype == np.uint8


def test_draw_detections_returns_same_size_and_draws_boxes():
    array = np.zeros((40, 60, 3), dtype=np.uint8)
    detections = [
        Detection("pizza", 0.9, (5, 5, 25, 25)),
        Detection("salad fresh", 0.8, (30, 10, 55, 35)),
    ]
    out = annotated_array(array, detections)
    assert out.shape == (40, 60, 3)
    assert out.dtype == np.uint8
    assert int(out.sum()) > 0  # something was drawn
    assert tuple(out[5, 5]) != (0, 0, 0)  # top-left corner of dish 1's box


def test_draw_detections_handles_degenerate_and_reversed_boxes():
    array = np.zeros((20, 20, 3), dtype=np.uint8)
    detections = [
        Detection("x", 0.5, (10, 10, 10, 10)),  # degenerate -> skipped
        Detection("y", 0.5, (15, 15, 3, 3)),  # reversed -> reordered
    ]
    out = annotated_array(array, detections)
    assert out.shape == (20, 20, 3)
    assert isinstance(draw_detections(array, detections), object)
