"""Automatic weight estimation using calibrated monocular depth.

MiDaS produces *relative* depth. Without a physical calibration the depth map
cannot be converted to centimetres, so this module checks calibration **before**
loading or running the depth model and reports an explicit ``unavailable``
result when the required values are missing. Nothing is assumed or invented.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .depth_model import (
    DepthModel,
    is_available as depth_model_available,
    load_depth_model,
    predict_depth,
)
from .estimator import DepthEvidence, WeightEstimator
from .volume import height_map_from_depth, integrate_volume_cm3


@dataclass
class AutoWeightConfig:
    """Physical calibration for the depth -> volume -> mass path."""

    cm_per_pixel: Optional[float] = None
    cm_per_depth_unit: Optional[float] = None
    reference_depth: Optional[float] = None
    invert_depth: bool = False

    @classmethod
    def from_weight_config(cls, weight_config) -> "AutoWeightConfig":
        """Build from an ``AppConfig.weight`` (or compatible object)."""
        return cls(
            cm_per_pixel=getattr(weight_config, "cm_per_pixel", None),
            cm_per_depth_unit=getattr(weight_config, "cm_per_depth_unit", None),
            reference_depth=getattr(weight_config, "reference_depth", None),
            invert_depth=bool(getattr(weight_config, "invert_depth", False)),
        )

    def calibration_error(self) -> Optional[str]:
        """Return a human-readable reason when calibration is unusable."""
        cpp = self.cm_per_pixel
        if cpp is None:
            return (
                "no physical scale calibration available (cm_per_pixel missing); "
                "relative MiDaS depth cannot be converted to centimetres reliably"
            )
        if not isinstance(cpp, (int, float)) or not math.isfinite(float(cpp)) or float(cpp) <= 0:
            return "invalid physical scale calibration (cm_per_pixel must be positive and finite)"
        cpd = self.cm_per_depth_unit
        if cpd is None:
            return (
                "no depth-scale calibration available (cm_per_depth_unit missing); "
                "cannot compute metric volume from relative depth"
            )
        if not isinstance(cpd, (int, float)) or not math.isfinite(float(cpd)) or float(cpd) <= 0:
            return (
                "invalid depth-scale calibration "
                "(cm_per_depth_unit must be positive and finite)"
            )
        if self.reference_depth is not None:
            if (
                not isinstance(self.reference_depth, (int, float))
                or not math.isfinite(float(self.reference_depth))
            ):
                return "invalid reference_depth (must be a finite number)"
        return None


class MiDaSDepthProvider:
    """Turn a food crop into depth evidence using MiDaS + explicit calibration.

    ``bundle`` may be injected for deterministic tests; otherwise the model is
    loaded lazily on the first call (never during configuration checks).
    """

    def __init__(
        self,
        weight_estimator: WeightEstimator,
        config: Optional[AutoWeightConfig] = None,
        bundle: Optional[DepthModel] = None,
        model_name: str = "midas_small",
    ) -> None:
        self.weight_estimator = weight_estimator
        self.cfg = config or AutoWeightConfig()
        self.model_name = model_name
        self._bundle = bundle
        self._load_error: Optional[str] = None

    def _ensure_loaded(self) -> None:
        if self._bundle is not None or self._load_error is not None:
            return
        try:
            if not depth_model_available():
                self._load_error = "torch/MiDaS is not available"
                return
            self._bundle = load_depth_model(self.model_name)
        except Exception as exc:  # noqa: BLE001 - reported as unavailable evidence
            self._load_error = str(exc)

    def estimate_depth_evidence(self, image_rgb, food_label: str = None, food_mask=None):
        # Calibration is checked first so we never download/run MiDaS when the
        # result could not be metric anyway.
        calibration_error = self.cfg.calibration_error()
        if calibration_error:
            return DepthEvidence(False, reason=calibration_error)

        self._ensure_loaded()
        if self._bundle is None:
            return DepthEvidence(
                False, reason=f"depth model unavailable: {self._load_error or 'not loaded'}"
            )

        try:
            array = np.asarray(image_rgb)
            if array.ndim != 3 or array.shape[-1] != 3:
                return DepthEvidence(False, reason="invalid image for depth estimation")
            if array.size == 0:
                return DepthEvidence(False, reason="empty image for depth estimation")
            depth_map = predict_depth(self._bundle, array)
        except Exception as exc:  # noqa: BLE001 - reported as unavailable evidence
            return DepthEvidence(False, reason=f"depth prediction failed: {exc}")

        depth = np.asarray(depth_map, dtype=np.float64)
        if depth.size == 0 or not np.any(np.isfinite(depth)):
            return DepthEvidence(False, reason="depth prediction returned no finite values")

        if food_mask is not None:
            mask = np.asarray(food_mask)
            if mask.shape != depth.shape:
                return DepthEvidence(
                    False,
                    reason="food mask shape does not match the depth map",
                )

        if self.cfg.reference_depth is None:
            reference_depth = float(np.nanmin(depth))
            reference_source = "inferred_min_depth"
        else:
            reference_depth = float(self.cfg.reference_depth)
            reference_source = "configured"

        try:
            height_map = height_map_from_depth(
                depth,
                reference_depth=reference_depth,
                invert=self.cfg.invert_depth,
                cm_per_depth_unit=float(self.cfg.cm_per_depth_unit),
            )
            volume_cm3 = integrate_volume_cm3(
                height_map, float(self.cfg.cm_per_pixel), food_mask=food_mask
            )
        except Exception as exc:  # noqa: BLE001 - reported as unavailable evidence
            return DepthEvidence(False, reason=f"volume integration failed: {exc}")

        if volume_cm3 is None or not math.isfinite(float(volume_cm3)) or volume_cm3 <= 0:
            return DepthEvidence(False, volume_cm3=volume_cm3, reason="computed volume is not usable")

        evidence = self.weight_estimator.depth_evidence_from(float(volume_cm3), food_label)
        if not evidence.available:
            return evidence

        calibration = {
            "cm_per_pixel": float(self.cfg.cm_per_pixel),
            "cm_per_depth_unit": float(self.cfg.cm_per_depth_unit),
            "reference_depth": reference_depth,
            "reference_depth_source": reference_source,
            "invert_depth": bool(self.cfg.invert_depth),
        }
        evidence.calibration = calibration
        if reference_source == "inferred_min_depth":
            evidence.reason = (
                "reference plane inferred from the minimum crop depth; supply "
                "reference_depth for a more reliable calibration"
            )
        return evidence

    def __call__(self, image_rgb, food_label=None, food_mask=None):
        return self.estimate_depth_evidence(image_rgb, food_label, food_mask)


def build_depth_provider(config, weight_estimator: WeightEstimator) -> MiDaSDepthProvider:
    """Build a provider from an ``AppConfig`` (or a ``WeightConfig``)."""
    weight_config = getattr(config, "weight", config)
    aw_cfg = AutoWeightConfig.from_weight_config(weight_config)
    # Try to load validated saved calibration if not explicitly set
    try:
        if aw_cfg.cm_per_pixel is None or aw_cfg.cm_per_depth_unit is None:
            from .calibration_fit import load_calibration

            saved = load_calibration()
            if isinstance(saved, dict) and saved.get("validated") is True:
                if aw_cfg.cm_per_pixel is None and saved.get("cm_per_pixel") is not None:
                    aw_cfg.cm_per_pixel = saved.get("cm_per_pixel")
                if aw_cfg.cm_per_depth_unit is None and saved.get("cm_per_depth_unit") is not None:
                    aw_cfg.cm_per_depth_unit = saved.get("cm_per_depth_unit")
                if aw_cfg.reference_depth is None and saved.get("reference_depth") is not None:
                    aw_cfg.reference_depth = saved.get("reference_depth")
                if saved.get("invert_depth") is not None:
                    aw_cfg.invert_depth = bool(saved.get("invert_depth"))
    except Exception:
        pass
    return MiDaSDepthProvider(
        weight_estimator,
        aw_cfg,
    )
