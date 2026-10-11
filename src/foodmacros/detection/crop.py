"""Full-resolution image loading and bounding-box cropping.

Used by the multi-dish pipeline: a detector returns boxes in the coordinate
space of the *full* image, so we must crop from the full-resolution pixels
(not a model-resized copy) before classifying each dish.

Boxes are pixel coordinates ``(x1, y1, x2, y2)``. Coordinates are clamped to
the image and an optional fractional padding expands each box so a little
context around the dish is kept. Nothing here invents data: an out-of-range or
degenerate box raises a clear error instead of returning a bad crop.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

from ..data.preprocessing import ImageDecodeError

Box = Tuple[float, float, float, float]


def load_full_image(path: str | Path) -> np.ndarray:
    """Load an image at its native resolution as an RGB ``uint8`` array.

    Unlike :func:`foodmacros.data.preprocessing.load_image`, the image is *not*
    resized, so detector pixel boxes can be applied directly.
    """
    from PIL import Image

    try:
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)
    except Exception as exc:  # noqa: BLE001 - re-raised with the path
        raise ImageDecodeError(f"Could not decode image '{path}': {exc}") from exc


def box_area(box: Sequence[float]) -> float:
    """Area of ``(x1, y1, x2, y2)`` in pixels (0 for a degenerate box)."""
    x1, y1, x2, y2 = (float(v) for v in box)
    return abs(x2 - x1) * abs(y2 - y1)


def clamp_box(
    box: Sequence[float],
    width: int,
    height: int,
    padding: float = 0.0,
) -> Tuple[int, int, int, int]:
    """Clamp a box to the image and expand it by ``padding`` of its size.

    ``padding`` is a fraction of the box's width/height (e.g. ``0.08`` adds 8%
    on each side). Returns integer pixel bounds ``(x1, y1, x2, y2)``.
    """
    if len(tuple(box)) != 4:
        raise ValueError(f"box must have 4 coordinates, got {box!r}")
    x1, y1, x2, y2 = (float(v) for v in box)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    pad = float(padding)
    if pad < 0:
        pad = 0.0
    px = (x2 - x1) * pad
    py = (y2 - y1) * pad

    ix1 = max(0, int(math.floor(x1 - px)))
    iy1 = max(0, int(math.floor(y1 - py)))
    ix2 = min(int(width), int(math.ceil(x2 + px)))
    iy2 = min(int(height), int(math.ceil(y2 + py)))
    return ix1, iy1, ix2, iy2


def crop_array(array: np.ndarray, box: Sequence[float], padding: float = 0.0) -> np.ndarray:
    """Return the pixels of ``array`` inside ``box`` (with optional padding).

    Raises :class:`ValueError` for a degenerate/empty crop so the caller can
    report the offending detection rather than silently classifying noise.
    """
    arr = np.asarray(array)
    if arr.ndim < 2:
        raise ValueError(f"expected an image array, got shape {arr.shape}")
    height, width = int(arr.shape[0]), int(arr.shape[1])
    x1, y1, x2, y2 = clamp_box(box, width, height, padding)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"box {tuple(box)} produces an empty crop within a {width}x{height} image"
        )
    return arr[y1:y2, x1:x2]
