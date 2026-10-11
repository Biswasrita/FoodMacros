"""Weight estimation: Nutrition5k regression, experimental depth, estimator."""

from .calibration import Calibration, calibration_from_plate, cm_per_pixel_from_known_size
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
    "Calibration",
    "calibration_from_plate",
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
    "find_metadata_files",
    "load_dish_metadata",
    "normalize_targets",
    "predict_mass",
    "train_mass_regressor",
    "height_map_from_depth",
    "integrate_volume_cm3",
    "pixel_area_cm2",
]
