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

# A modern high-contrast vibrant palette matching the Vision-AI UI.
_COLORS = [
    (16, 185, 129),   # Emerald
    (6, 182, 212),    # Cyan
    (245, 158, 11),   # Amber
    (139, 92, 246),   # Purple
    (244, 63, 94),    # Rose
    (59, 130, 246),   # Blue
    (234, 179, 8),    # Yellow
    (168, 85, 247),   # Violet
    (20, 184, 166),   # Teal
    (239, 68, 68),    # Red
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
    from PIL import Image, ImageDraw

    base = _to_pil(image)
    width, height = base.size
    
    # Create semi-transparent overlay for food area illumination
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw_ov = ImageDraw.Draw(overlay)
    draw = ImageDraw.Draw(base)

    line_w = max(3, int(round(min(width, height) * 0.009)))
    font = _load_font(max(15, int(round(min(width, height) * 0.038))))

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
        fill_color = color + (35,)  # Subtle 14% translucent tint

        # Draw tinted fill in overlay
        draw_ov.rectangle([x1, y1, x2, y2], fill=fill_color)
        
        # Draw solid prominent bounding border directly on base
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w)

        caption = labels[i - 1] if labels and i - 1 < len(labels) else f"Dish {i}"
        try:
            tb = draw.textbbox((0, 0), caption, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        except Exception:
            tw, th = (7 * len(caption), 12)
        ty = y1 - th - 6
        if ty < 0:
            ty = y1 + 4
        # Draw badge background
        draw.rectangle([x1, ty, x1 + tw + 10, ty + th + 6], fill=color)
        draw.text((x1 + 5, ty + 2), caption, fill=(255, 255, 255), font=font)

    # Composite translucent highlight over base image
    base = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
    return base


def annotated_array(image, detections: Sequence, labels: Sequence[str] | None = None) -> np.ndarray:
    """Convenience: :func:`draw_detections` returned as an RGB ``uint8`` array."""
    return np.asarray(draw_detections(image, detections, labels=labels), dtype=np.uint8)
