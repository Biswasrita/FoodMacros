"""Stage 5 tests for calibration and volume geometry (no TF required)."""

import numpy as np
import pytest

from foodmacros.weight.calibration import (
    Calibration,
    calibration_from_plate,
    cm_per_pixel_from_known_size,
)
from foodmacros.weight.density import DensityTable
from foodmacros.weight.volume import (
    height_map_from_depth,
    integrate_volume_cm3,
    pixel_area_cm2,
)


def test_cm_per_pixel_from_known_size():
    assert cm_per_pixel_from_known_size(20.0, 400.0) == pytest.approx(0.05)


def test_cm_per_pixel_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        cm_per_pixel_from_known_size(20.0, 0)
    with pytest.raises(ValueError):
        cm_per_pixel_from_known_size(0.0, 100.0)


def test_calibration_from_plate_valid():
    calib = calibration_from_plate(25.0, 500.0, cm_per_depth_unit=0.1)
    assert calib.valid
    assert calib.cm_per_pixel == pytest.approx(0.05)
    assert calib.cm_per_depth_unit == pytest.approx(0.1)


def test_calibration_rejects_bad_depth_unit():
    with pytest.raises(ValueError):
        calibration_from_plate(25.0, 500.0, cm_per_depth_unit=-1)


def test_calibration_dataclass_validity():
    assert not Calibration(cm_per_pixel=0).valid
    assert Calibration(cm_per_pixel=0.05).valid


def test_pixel_area_cm2():
    assert pixel_area_cm2(0.5) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        pixel_area_cm2(0)


def test_height_map_from_depth_and_invert():
    depth = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    h = height_map_from_depth(depth, reference_depth=1.0, cm_per_depth_unit=1.0)
    assert h.tolist() == [[0, 1, 2], [3, 4, 5]]
    inv = height_map_from_depth(depth, reference_depth=1.0, invert=True)
    assert inv.sum() == 0


def test_height_map_scales_by_depth_unit():
    depth = np.array([[2.0, 3.0]], dtype=float)
    h = height_map_from_depth(depth, reference_depth=2.0, cm_per_depth_unit=2.0)
    assert h.tolist() == [[0.0, 2.0]]


def test_integrate_volume_cm3():
    heights = np.array([[0, 1, 2], [3, 4, 5]], dtype=float)
    assert integrate_volume_cm3(heights, cm_per_pixel=0.5) == pytest.approx(3.75)


def test_integrate_volume_with_mask():
    heights = np.array([[0, 1, 2], [3, 4, 5]], dtype=float)
    mask = np.array([[1, 0, 0], [0, 1, 0]], dtype=bool)
    # kept pixels: (0,0)=0 and (1,1)=4 -> sum 4 -> 4 * 0.5^2 = 1.0
    assert integrate_volume_cm3(heights, 0.5, food_mask=mask) == pytest.approx(1.0)


def test_integrate_volume_mask_shape_mismatch():
    with pytest.raises(ValueError):
        integrate_volume_cm3(np.zeros((2, 2)), 0.5, food_mask=np.zeros((3, 3)))


def test_density_table_default_and_case_insensitive():
    table = DensityTable.from_json()
    assert len(table) > 0
    assert table.get("PIZZA") is not None
    assert table.get("definitely_not_a_food") is None


def test_density_table_bad_entry_raises(tmp_path):
    import json

    bad = tmp_path / "d.json"
    bad.write_text(json.dumps({"densities": {"x": {"note": "no value"}}}), encoding="utf-8")
    with pytest.raises(ValueError):
        DensityTable.from_json(bad)
