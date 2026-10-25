"""Fit and validate depth calibration from REAL measurements only.

No synthetic/fabricated calibration values are generated here. Fitting requires
actual reference measurements and actual measured food weights. Results must
be validated on held-out samples.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..config import PROJECT_ROOT
from .calibration import cm_per_pixel_from_known_size, cm_per_depth_unit_from_reference
from .calibration_dataset import (
    CalibrationSample,
    DEFAULT_CALIBRATED_PARAMS,
    DEFAULT_CALIBRATION_REPORT,
    load_calibration_samples,
    validate_sample,
)

try:
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None


@dataclass
class FitResult:
    success: bool
    calibrated_params: Dict[str, Any]
    report: Dict[str, Any]
    errors: List[str]
    warnings: List[str]


def _get_known_size_cm_px(ref) -> Tuple[Optional[float], Optional[float]]:
    try:
        known_cm = ref.known_cm or {}
        pixels = ref.pixels or {}
        cm = known_cm.get("width") or known_cm.get("diameter") or known_cm.get("height") or known_cm.get("size")
        px = pixels.get("width") or pixels.get("diameter") or pixels.get("height") or pixels.get("size")
        if cm is None or px is None:
            # try combined
            if "width_cm" in known_cm and "width_px" in pixels:
                cm, px = known_cm["width_cm"], pixels["width_px"]
        return (float(cm) if cm is not None else None, float(px) if px is not None else None)
    except Exception:
        return (None, None)


def _linear_fit(x: List[float], y: List[float]) -> Tuple[float, float, Dict[str, float]]:
    """Fit y = a*x + b; return (a,b,metrics)."""
    if _np is not None and len(x) >= 2:
        X = _np.array(x, dtype=float)
        Y = _np.array(y, dtype=float)
        A = _np.vstack([X, _np.ones(len(X))]).T
        a, b = _np.linalg.lstsq(A, Y, rcond=None)[0]
        pred = a * X + b
        resid = Y - pred
        mse = float(_np.mean(resid ** 2))
        rmse = math.sqrt(mse)
        mae = float(_np.mean(_np.abs(resid)))
        ss_res = float(_np.sum(resid ** 2))
        ss_tot = float(_np.sum((Y - _np.mean(Y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-18 else float("nan")
        return float(a), float(b), {"rmse": rmse, "mae": mae, "r2": r2}
    # fallback
    if len(x) < 2:
        raise ValueError("need >=2 points for linear fit")
    # simple
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den = sum((xi - mx) ** 2 for xi in x)
    a = num / den if den != 0 else 0.0
    b = my - a * mx
    pred = [a * xi + b for xi in x]
    resid = [yi - pi for yi, pi in zip(y, pred)]
    mae = sum(abs(r) for r in resid) / len(resid)
    mse = sum(r * r for r in resid) / len(resid)
    rmse = math.sqrt(mse)
    return a, b, {"rmse": rmse, "mae": mae, "r2": float("nan")}


def _evaluate(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
    if not y_true or len(y_true) != len(y_pred):
        return {"mae": float("nan"), "rmse": float("nan"), "mape_pct": float("nan")}
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
    return {"mae": mae, "rmse": rmse, "mape_pct": mape_pct}


def fit_calibration_from_samples(samples: List[CalibrationSample]) -> FitResult:
    errors: List[str] = []
    warnings: List[str] = []

    if not samples:
        return FitResult(False, {}, {"status": "no_samples"}, ["No calibration samples provided. Collect real photos with known references and measured weights."], [])

    # Validate all samples
    for s in samples:
        errs = validate_sample(s)
        for e in errs:
            errors.append(f"{s.id}: {e}")

    if errors:
        return FitResult(False, {}, {"status": "validation_failed"}, errors, warnings)

    # Compute cm_per_pixel per sample
    cm_per_pixel_values = []
    for s in samples:
        cpp = None
        for r in s.references:
            if r.type == "known_size":
                cm, px = _get_known_size_cm_px(r)
                if cm and px and px > 0:
                    cpp = cm_per_pixel_from_known_size(cm, px)
                    break
        if cpp is None or not math.isfinite(cpp) or cpp <= 0:
            errors.append(f"{s.id}: could not compute valid cm_per_pixel from references")
            continue
        cm_per_pixel_values.append((s, cpp))

    if errors:
        return FitResult(False, {}, {"status": "calibration_failed"}, errors, warnings)

    # Try to fit depth scale if depth stats available
    train_samples = [s for s in samples if s.split == "train"]
    val_samples = [s for s in samples if s.split == "val"]
    holdout_samples = [s for s in samples if s.split == "holdout"]

    if not train_samples:
        warnings.append("No samples marked 'train'; using all samples for fitting but this is not ideal. Keep train/val/holdout separate.")

    fit_set = train_samples if train_samples else samples

    # Check if we can fit depth scale
    depth_fit_points = []
    for s in fit_set:
        ds = s.depth_stats or {}
        d_plane = ds.get("plane_depth") or ds.get("reference_depth")
        d_obj = ds.get("object_top_depth") or ds.get("max_depth")
        known_h = None
        for r in s.references:
            if r.type == "known_height":
                kh = r.known_cm or {}
                known_h = kh.get("height") or kh.get("height_cm")
                break
        if known_h is not None and d_plane is not None and d_obj is not None:
            try:
                known_h_f = float(known_h)
                d_plane_f = float(d_plane)
                d_obj_f = float(d_obj)
                diff = abs(d_obj_f - d_plane_f)
                if diff > 1e-12:
                    depth_fit_points.append((diff, known_h_f))
            except Exception:
                continue

    calibrated_params: Dict[str, Any] = {
        "version": 1,
        "fit_method": None,
        "cm_per_pixel": None,
        "cm_per_depth_unit": None,
        "reference_depth": None,
        "invert_depth": False,
        "validated": False,
    }

    report: Dict[str, Any] = {
        "status": "insufficient_data",
        "num_samples": len(samples),
        "train": len(train_samples),
        "val": len(val_samples),
        "holdout": len(holdout_samples),
        "metrics": {},
        "notes": [],
    }

    # Use median/mean cm_per_pixel as initial estimate (must come from real data)
    cpp_list = [cpp for _, cpp in cm_per_pixel_values]
    if cpp_list:
        cpp_list_sorted = sorted(cpp_list)
        median_cpp = cpp_list_sorted[len(cpp_list_sorted) // 2]
        calibrated_params["cm_per_pixel"] = float(median_cpp)
        report["cm_per_pixel_median"] = float(median_cpp)
        report["cm_per_pixel_values"] = cpp_list

    if len(depth_fit_points) >= 3:
        try:
            x = [p[0] for p in depth_fit_points]
            y = [p[1] for p in depth_fit_points]
            a, b, fit_metrics = _linear_fit(x, y)
            # Prefer model through origin if intercept near zero; else use linear
            if abs(b) < 0.1 * (abs(a) + 1e-12):  # small intercept
                cm_per_depth_unit = float(a) if a > 0 else None
                calibrated_params["cm_per_depth_unit"] = cm_per_depth_unit
                calibrated_params["fit_method"] = "linear_through_origin_from_known_height_references"
                calibrated_params["fit_intercept"] = float(b)
            else:
                # Linear model y = a*x + b; cm_per_depth_unit as slope is reasonable if validated
                cm_per_depth_unit = float(a) if a > 0 else None
                calibrated_params["cm_per_depth_unit"] = cm_per_depth_unit
                calibrated_params["fit_method"] = "linear_from_known_height_references"
                calibrated_params["fit_intercept"] = float(b)
            report["depth_fit"] = {"num_points": len(depth_fit_points), "slope": float(a), "intercept": float(b), **fit_metrics}
        except Exception as e:
            warnings.append(f"Failed to fit depth scale: {e}")
    else:
        warnings.append(
            "Insufficient known-height reference samples with depth stats to fit depth scale reliably. "
            "Need >=3 samples with known_height references and plane/object depth measurements."
        )

    # Validation on held-out if possible
    if holdout_samples and calibrated_params.get("cm_per_pixel") and calibrated_params.get("cm_per_depth_unit"):
        # This is a placeholder - full volume estimation requires running MiDaS; we validate calibration params only
        report["status"] = "fitted_but_not_fully_evaluated_volume_estimation"
        report["notes"].append("Calibration parameters fitted from references, but end-to-end volume estimation not validated without running depth on held-out samples.")
        calibrated_params["validated"] = False  # Not validated until end-to-end evaluation with real measurements
    elif calibrated_params.get("cm_per_pixel") and calibrated_params.get("cm_per_depth_unit"):
        report["status"] = "fitted_missing_holdout"
        report["notes"].append("No held-out samples - cannot validate on unseen data. Split samples into train/val/holdout.")
        calibrated_params["validated"] = False
    else:
        report["status"] = "insufficient_calibration"
        calibrated_params["validated"] = False

    if calibrated_params["validated"]:
        report["status"] = "validated"
    else:
        # Never claim validated without evidence
        calibrated_params["validated"] = False

    return FitResult(True, calibrated_params, report, errors, warnings)


def save_calibration(calibrated_params: Dict[str, Any], report: Dict[str, Any], params_path: str | Path | None = None, report_path: str | Path | None = None) -> Tuple[Path, Path]:
    pp = Path(params_path) if params_path is not None else DEFAULT_CALIBRATED_PARAMS
    rp = Path(report_path) if report_path is not None else DEFAULT_CALIBRATION_REPORT
    pp.parent.mkdir(parents=True, exist_ok=True)
    rp.parent.mkdir(parents=True, exist_ok=True)
    pp.write_text(json.dumps(calibrated_params, indent=2), encoding="utf-8")
    rp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return pp, rp


def load_calibration(params_path: str | Path | None = None) -> Optional[Dict[str, Any]]:
    pp = Path(params_path) if params_path is not None else DEFAULT_CALIBRATED_PARAMS
    if not pp.exists():
        return None
    try:
        return json.loads(pp.read_text(encoding="utf-8"))
    except Exception:
        return None
