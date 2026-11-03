"""Weight estimation: Nutrition5k regression, experimental depth, estimator."""

from .calibration import Calibration, calibration_from_plate, cm_per_pixel_from_known_size
from .density import DensityEntry, DensityTable
from .depth_model import DepthModel, is_available as depth_available, load_depth_model, predict_depth
from .estimator import DepthEvidence, WeightEstimate, WeightEstimator
from .mass_dataset import (
    MassDatasetRecord,
    build_weight_dataset,
    build_weight_dataset_csv,
    find_nutrition5k_metadata,
    load_nutrition5k_mass_metadata,
    split_by_dish_id,
)
from .regression import (
    MASS_ONLY_TARGETS,
    TARGET_COLUMNS,
    MassPrediction,
    RegressionRecord,
    build_mass_only_regressor,
    build_regression_model,
    build_regression_records,
    compute_norm_stats,
    denormalize_targets,
    find_metadata_files,
    get_regression_preprocess_input,
    load_dish_metadata,
    normalize_targets,
    predict_mass,
    predict_mass_only_from_array,
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
    "MASS_ONLY_TARGETS",
    "TARGET_COLUMNS",
    "MassPrediction",
    "RegressionRecord",
    "MassDatasetRecord",
    "build_mass_only_regressor",
    "build_regression_model",
    "build_regression_records",
    "build_weight_dataset",
    "build_weight_dataset_csv",
    "compute_norm_stats",
    "denormalize_targets",
    "find_metadata_files",
    "find_nutrition5k_metadata",
    "get_regression_preprocess_input",
    "load_dish_metadata",
    "load_nutrition5k_mass_metadata",
    "normalize_targets",
    "predict_mass",
    "predict_mass_only_from_array",
    "split_by_dish_id",
    "train_mass_regressor",
    "height_map_from_depth",
    "integrate_volume_cm3",
    "pixel_area_cm2",
]
