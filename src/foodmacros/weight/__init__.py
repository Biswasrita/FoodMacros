"""Weight estimation: Nutrition5k regression, experimental depth, estimator."""

from .auto_weight import AutoWeightConfig, MiDaSDepthProvider, build_depth_provider
from .calibration import (
    Calibration,
    calibration_from_plate,
    cm_per_depth_unit_from_reference,
    cm_per_pixel_from_known_size,
)
from .calibration_dataset import (
    CalibrationReference,
    CalibrationSample,
    load_calibration_samples,
    save_calibration_samples,
    split_samples,
    validate_sample,
)
from .calibration_eval import evaluate_estimates
from .calibration_fit import fit_calibration_from_samples, load_calibration, save_calibration
from .density import DensityEntry, DensityTable
from .depth_model import DepthModel, is_available as depth_available, load_depth_model, predict_depth
from .estimator import DepthEvidence, WeightEstimate, WeightEstimator
from .regression import (
    TARGET_COLUMNS,
    MassPrediction,
    RegressionRecord,
    build_regression_model,
    build_regression_records,
    compute_norm_stats,
    denormalize_targets,
    find_metadata_files,
    load_dish_metadata,
    normalize_targets,
    predict_mass,
    train_mass_regressor,
)
from .volume import height_map_from_depth, integrate_volume_cm3, pixel_area_cm2

__all__ = [
    "AutoWeightConfig",
    "MiDaSDepthProvider",
    "build_depth_provider",
    "Calibration",
    "CalibrationReference",
    "CalibrationSample",
    "calibration_from_plate",
    "cm_per_depth_unit_from_reference",
    "cm_per_pixel_from_known_size",
    "DensityEntry",
    "DensityTable",
    "DepthModel",
    "depth_available",
    "load_depth_model",
    "predict_depth",
    "DepthEvidence",
    "WeightEstimate",
    "WeightEstimator",
    "TARGET_COLUMNS",
    "MassPrediction",
    "RegressionRecord",
    "build_regression_model",
    "build_regression_records",
    "compute_norm_stats",
    "denormalize_targets",
    "evaluate_estimates",
    "find_metadata_files",
    "fit_calibration_from_samples",
    "load_calibration",
    "load_calibration_samples",
    "load_dish_metadata",
    "normalize_targets",
    "predict_mass",
    "save_calibration",
    "save_calibration_samples",
    "split_samples",
    "train_mass_regressor",
    "validate_sample",
    "height_map_from_depth",
    "integrate_volume_cm3",
    "pixel_area_cm2",
]
