"""Calibration and geometric helpers for the experimental depth path.

To turn depth + a food mask into a metric volume you need two scales:
  * ``cm_per_pixel``      - image-plane scale (from a known-size reference),
  * ``cm_per_depth_unit`` - how many cm one unit of the depth map represents.

Both must come from a real reference; neither is assumed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Calibration:
    cm_per_pixel: float
    cm_per_depth_unit: float | None = None
    reference: str = ""

    @property
    def valid(self) -> bool:
        return self.cm_per_pixel is not None and self.cm_per_pixel > 0


def cm_per_pixel_from_known_size(known_size_cm: float, known_size_pixels: float) -> float:
    """Image-plane scale from a reference of known real size."""
    if known_size_pixels is None or known_size_pixels <= 0:
        raise ValueError("known_size_pixels must be positive.")
    if known_size_cm is None or known_size_cm <= 0:
        raise ValueError("known_size_cm must be positive.")
    return float(known_size_cm) / float(known_size_pixels)


def calibration_from_plate(
    plate_diameter_cm: float,
    plate_diameter_pixels: float,
    cm_per_depth_unit: float | None = None,
) -> Calibration:
    """Calibrate using a plate (or any circular reference) of known diameter."""
    cpp = cm_per_pixel_from_known_size(plate_diameter_cm, plate_diameter_pixels)
    if cm_per_depth_unit is not None and cm_per_depth_unit <= 0:
        raise ValueError("cm_per_depth_unit must be positive when provided.")
    return Calibration(
        cm_per_pixel=cpp,
        cm_per_depth_unit=cm_per_depth_unit,
        reference=f"plate diameter {plate_diameter_cm} cm / {plate_diameter_pixels} px",
    )
