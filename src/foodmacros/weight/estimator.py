"""Portion-weight estimation with explicit, honest status handling.

Sources, in order of trust:
  1. ``measured``     - a weight the user actually measured.
  2. ``estimated``    - a validated Nutrition5k regression prediction.
  3. ``experimental``  - depth-volume x approximate density (off by default).
  4. ``unavailable``  - nothing reliable; **no number is invented**.

A gram value is only ever produced by a source that actually produced one,
and only when it falls inside the configured reliable range.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..config import AppConfig
from .density import DensityTable

VALID_STATUSES = ("measured", "estimated", "experimental", "unavailable")


@dataclass
class DepthEvidence:
    available: bool
    volume_cm3: Optional[float] = None
    density_g_cm3: Optional[float] = None
    reason: str = ""
    density_approximate: bool = True
    calibration: Optional[Dict[str, object]] = None


@dataclass
class WeightEstimate:
    status: str
    grams: Optional[float]
    method: str
    confidence: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, object] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.status != "unavailable"

    def summary(self) -> str:
        if not self.available:
            reason = "; ".join(self.warnings) or "no weight source"
            return f"weight unavailable ({self.method}): {reason}"
        return f"{self.grams:.1f} g [{self.status} via {self.method}]"


class WeightEstimator:
    def __init__(self, config: Optional[AppConfig] = None, density_table: Optional[DensityTable] = None):
        self.config = config or AppConfig()
        self.weight_cfg = self.config.weight
        self.density_table = density_table if density_table is not None else DensityTable.from_json()

    def _reliable(self, grams: Optional[float]) -> bool:
        if grams is None or not math.isfinite(float(grams)):
            return False
        return self.weight_cfg.min_reliable_grams <= grams <= self.weight_cfg.max_reliable_grams

    @staticmethod
    def _depth_grams(evidence: DepthEvidence) -> Optional[float]:
        if evidence.volume_cm3 is None or evidence.density_g_cm3 is None:
            return None
        if evidence.volume_cm3 <= 0 or evidence.density_g_cm3 <= 0:
            return None
        return float(evidence.volume_cm3) * float(evidence.density_g_cm3)

    def depth_evidence_from(self, volume_cm3: float, food_label: str) -> DepthEvidence:
        """Build depth evidence for a label, using the density table if known."""
        if volume_cm3 is None or volume_cm3 <= 0:
            return DepthEvidence(False, volume_cm3=volume_cm3, reason="no metric volume available")
        entry = self.density_table.get(food_label) if self.density_table is not None else None
        if entry is None:
            return DepthEvidence(
                False, volume_cm3=volume_cm3,
                reason=f"no density entry for '{food_label}' (add one to enable the depth path)",
            )
        return DepthEvidence(
            True,
            volume_cm3=volume_cm3,
            density_g_cm3=entry.density_g_cm3,
            density_approximate=bool(entry.approximate),
        )

    def estimate(
        self,
        measured_grams: Optional[float] = None,
        regression=None,
        depth: Optional[DepthEvidence] = None,
        food_label: Optional[str] = None,
    ) -> WeightEstimate:
        warnings: List[str] = []

        # 1) User-measured weight is the most trustworthy.
        if measured_grams is not None:
            try:
                grams = float(measured_grams)
            except (TypeError, ValueError):
                return WeightEstimate(
                    "unavailable", None, "user-provided", None,
                    [f"measured weight '{measured_grams}' is not a number"],
                )
            if not math.isfinite(grams) or grams <= 0:
                return WeightEstimate(
                    "unavailable", None, "user-provided", None,
                    ["measured weight must be a positive, finite number"],
                )
            if not self._reliable(grams):
                warnings.append(
                    f"measured weight {grams:.1f} g is outside the configured "
                    f"reliable range [{self.weight_cfg.min_reliable_grams}, "
                    f"{self.weight_cfg.max_reliable_grams}] g"
                )
            return WeightEstimate("measured", grams, "user-provided", "high", warnings)

        # 2) Nutrition5k regression.
        if regression is not None and getattr(regression, "available", False):
            grams = getattr(regression, "total_mass_g", None)
            if grams is None or not math.isfinite(float(grams)):
                warnings.append("regression returned a non-finite mass")
            elif self._reliable(grams):
                details = {
                    "calories": getattr(regression, "calories", None),
                    "fat_g": getattr(regression, "fat_g", None),
                    "carb_g": getattr(regression, "carb_g", None),
                    "protein_g": getattr(regression, "protein_g", None),
                }
                return WeightEstimate(
                    "estimated", float(grams), "nutrition5k-regression", "medium", warnings, details
                )
            else:
                warnings.append(
                    f"regression mass {float(grams):.1f} g is outside the reliable range "
                    f"[{self.weight_cfg.min_reliable_grams}, {self.weight_cfg.max_reliable_grams}] g"
                )

        # 3) Experimental depth-volume x density.
        if depth is not None and depth.available:
            if not self.weight_cfg.allow_experimental_depth:
                warnings.append("depth evidence present but the experimental depth path is disabled")
            else:
                grams = self._depth_grams(depth)
                if grams is None:
                    warnings.append("depth evidence insufficient (missing metric volume or density)")
                elif self._reliable(grams):
                    warnings_for_estimate = list(warnings)
                    if depth.density_approximate:
                        warnings_for_estimate.append(
                            "experimental estimate: relies on an approximate bulk density; "
                            "it is not a measurement"
                        )
                    return WeightEstimate(
                        "experimental", grams, "depth-volume-density", "low",
                        warnings_for_estimate + [
                            "experimental estimate: relies on a physical calibration "
                            "reference; it is not a measurement",
                        ],
                        {
                            "volume_cm3": depth.volume_cm3,
                            "density_g_cm3": depth.density_g_cm3,
                            "density_approximate": depth.density_approximate,
                            "calibration": depth.calibration,
                        },
                    )
                else:
                    warnings.append(
                        f"depth estimate {grams:.1f} g is outside the reliable range "
                        f"[{self.weight_cfg.min_reliable_grams}, {self.weight_cfg.max_reliable_grams}] g"
                    )
        elif depth is not None and not depth.available and depth.reason:
            warnings.append(f"depth evidence unavailable: {depth.reason}")

        if not warnings:
            warnings.append("no weight source available (no measured value, regression model, or depth evidence)")
        return WeightEstimate("unavailable", None, "none", None, warnings)
