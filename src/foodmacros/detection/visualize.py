"""Draw detected food boxes with numbered labels onto an image.

Used by the multi-dish UI so a user can see *where* each detected dish is and
match it to the numbered rows in the dish table. Boxes are the exact boxes the
pipeline used for cropping; nothing here changes the analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .crop import load_full_image

# A small, high-contrast palette (repeats if there are more dishes than colours).
_COLORS = [
    (230, 25, 75),
    (60, 180, 75),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 190),
    (0, 128, 128),
]


def _to_pil(image):
    from PIL import Image

    if isinstance(image, (str, Path)):
        return Image.fromarray(load_full_image(image)).convert("RGB")
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"expected an HxWx3 RGB image, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def _load_font(size: int):
    from PIL import ImageFont

    for name in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def draw_detections(image, detections: Sequence, labels: Sequence[str] | None = None):
    """Return a copy of ``image`` with each detection boxed and numbered.

    ``detections`` may be objects with a ``box_xyxy`` attribute (the pipeline's
    ``Detection``) or raw ``(x1, y1, x2, y2)`` tuples. ``labels`` optionally
    overrides the default ``"Dish N"`` captions.
    """
    from PIL import ImageDraw

    base = _to_pil(image)
    width, height = base.size
    draw = ImageDraw.Draw(base)
    line_w = max(2, int(round(min(width, height) * 0.006)))
    font = _load_font(max(14, int(round(min(width, height) * 0.035))))

    for i, det in enumerate(detections, start=1):
        box = getattr(det, "box_xyxy", det)
        try:
            x1, y1, x2, y2 = (float(v) for v in box)
        except (TypeError, ValueError):
            continue
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        x1 = max(0, min(width - 1, int(round(x1))))
        y1 = max(0, min(height - 1, int(round(y1))))
        x2 = max(0, min(width - 1, int(round(x2))))
        y2 = max(0, min(height - 1, int(round(y2))))
        if x2 <= x1 or y2 <= y1:
            continue

        color = _COLORS[(i - 1) % len(_COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w)

        caption = labels[i - 1] if labels and i - 1 < len(labels) else f"Dish {i}"
        try:
            tb = draw.textbbox((0, 0), caption, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        except Exception:
            tw, th = (7 * len(caption), 12)
        ty = y1 - th - 4
        if ty < 0:
            ty = y1 + 2
        draw.rectangle([x1, ty, x1 + tw + 6, ty + th + 4], fill=color)
        draw.text((x1 + 3, ty + 2), caption, fill=(255, 255, 255), font=font)

    return base


def annotated_array(image, detections: Sequence, labels: Sequence[str] | None = None) -> np.ndarray:
    """Convenience: :func:`draw_detections` returned as an RGB ``uint8`` array."""
    return np.asarray(draw_detections(image, detections, labels=labels), dtype=np.uint8)
