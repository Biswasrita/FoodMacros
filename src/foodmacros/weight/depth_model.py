"""Optional monocular-depth model loader (experimental).

Uses a pretrained MiDaS model (via PyTorch) to produce a *relative* depth map.
Relative depth has no absolute scale, so it cannot yield grams without the
calibration in ``calibration.py`` plus a density. This module is a thin,
guarded wrapper: it raises clear errors when PyTorch or weights are missing
instead of fabricating depth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ..env_checks import has_package

SUPPORTED_DEPTH_MODELS = ("midas_small",)


def is_available() -> bool:
    return has_package("torch")


@dataclass
class DepthModel:
    model: Any
    transform: Any
    name: str


def load_depth_model(name: str = "midas_small") -> DepthModel:
    """Load a pretrained depth model. Requires the optional ``torch`` package."""
    if not has_package("torch"):
        raise RuntimeError(
            "Depth estimation requires the optional 'torch' package. Install it "
            "(pip install torch) and ensure pretrained weights are reachable, or "
            "keep the depth path disabled (weight.allow_experimental_depth=false)."
        )
    import torch  # noqa: F401

    if name == "midas_small":
        model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        model.eval()
        return DepthModel(model=model, transform=transforms.small_transform, name=name)
    raise ValueError(
        f"Unsupported depth model '{name}'. Supported: {list(SUPPORTED_DEPTH_MODELS)}."
    )


def predict_depth(bundle: DepthModel, image_rgb) -> np.ndarray:
    """Return a relative depth map (float32) for an RGB uint8 array."""
    if not has_package("torch"):
        raise RuntimeError("Depth estimation requires the optional 'torch' package.")
    import torch

    array = np.asarray(image_rgb)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("image_rgb must be an HxWx3 array.")
    with torch.no_grad():
        tensor = bundle.transform(array)
        prediction = bundle.model(tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=array.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    return prediction.cpu().numpy().astype(np.float32)
