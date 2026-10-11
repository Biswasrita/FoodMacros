"""Volume integration for the experimental depth path.

A depth map becomes a height field above a reference plane, which is then
integrated over the food mask to a volume in cm^3. All conversions require an
explicit calibration; nothing is hard-coded.
"""

from __future__ import annotations

import numpy as np


def pixel_area_cm2(cm_per_pixel: float) -> float:
    if cm_per_pixel is None or cm_per_pixel <= 0:
        raise ValueError("cm_per_pixel must be positive.")
    return float(cm_per_pixel) ** 2


def height_map_from_depth(
    depth_map,
    reference_depth: float | None = None,
    invert: bool = False,
    cm_per_depth_unit: float = 1.0,
) -> np.ndarray:
    """Convert a depth map to a height field (in cm) above a reference plane.

    Assumption (documented, not universal): larger depth values are *closer*
    to the camera unless ``invert`` is set. Values below the reference plane
    are clipped to zero. ``cm_per_depth_unit`` scales depth units to cm.
    """
    depth = np.asarray(depth_map, dtype=np.float64)
    if depth.size == 0:
        raise ValueError("depth_map is empty.")
    ref = float(np.nanmin(depth)) if reference_depth is None else float(reference_depth)
    signed = (depth - ref) if not invert else (ref - depth)
    height_units = np.clip(signed, 0.0, None)
    return height_units * float(cm_per_depth_unit)


def integrate_volume_cm3(height_map, cm_per_pixel: float, food_mask=None) -> float:
    """Integrate a height field (cm) over pixels to a volume in cm^3."""
    heights = np.asarray(height_map, dtype=np.float64)
    heights = np.nan_to_num(heights, nan=0.0, posinf=0.0, neginf=0.0)
    if food_mask is not None:
        mask = np.asarray(food_mask).astype(bool)
        if mask.shape != heights.shape:
            raise ValueError("food_mask shape must match height_map shape.")
        heights = np.where(mask, heights, 0.0)
    area = pixel_area_cm2(cm_per_pixel)
    return float(np.sum(np.clip(heights, 0.0, None)) * area)
