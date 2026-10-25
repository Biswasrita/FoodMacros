"""Evaluate calibrated estimates against real measured weights (held-out only)."""

from __future__ import annotations

import math
from typing import Any, Dict, List

from .calibration_dataset import CalibrationSample, validate_sample


def evaluate_estimates(samples: List[CalibrationSample], estimates_g: List[float]) -> Dict[str, Any]:
    if len(samples) != len(estimates_g):
        return {"error": "mismatched lengths"}
    # Only evaluate on holdout with valid measurements
    holdout = []
    for s, e in zip(samples, estimates_g):
        if s.split == "holdout" and s.measured_weight_g is not None:
            try:
                holdout.append((float(s.measured_weight_g), float(e)))
            except Exception:
                continue
    if not holdout:
        return {"status": "no_holdout_evaluated", "message": "No valid holdout samples with measurements found."}
    y_true = [h[0] for h in holdout]
    y_pred = [h[1] for h in holdout]
    n = len(y_true)
    mae = sum(abs(t - p) for t, p in zip(y_true, y_pred)) / n
    mse = sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / n
    rmse = math.sqrt(mse)
    mape = 0.0
    count = 0
    for t, p in zip(y_true, y_pred):
        if abs(t) > 1e-12:
            mape += abs((t - p) / t)
            count += 1
    mape_pct = (mape / count * 100.0) if count > 0 else float("nan")
    return {
        "status": "evaluated",
        "num_holdout": n,
        "mae_g": mae,
        "rmse_g": rmse,
        "mape_pct": mape_pct,
    }
