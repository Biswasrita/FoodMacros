"""Data-layer: discovery, splits and preprocessing."""

from .discovery import (
    IMAGE_EXTS,
    NUTRITION_REQUIRED_COLUMNS,
    ResolvedInputs,
    default_search_roots,
    discover_detection_weights,
    discover_nutrition5k_roots,
    discover_nutrition_csvs,
    find_classification_roots,
    inspect_csv_columns,
    resolve_inputs,
)
from .preprocessing import (
    ImageDecodeError,
    build_tf_dataset,
    dataset_from_pairs,
    is_decodable,
    load_image,
    preprocess_arrays,
    preprocess_image,
    safe_load_image,
)
from .splits import (
    SplitResult,
    build_class_names,
    list_classes,
    load_class_names,
    make_splits,
    save_class_names,
)

__all__ = [
    "IMAGE_EXTS",
    "NUTRITION_REQUIRED_COLUMNS",
    "ResolvedInputs",
    "default_search_roots",
    "discover_detection_weights",
    "discover_nutrition5k_roots",
    "discover_nutrition_csvs",
    "find_classification_roots",
    "inspect_csv_columns",
    "resolve_inputs",
    "ImageDecodeError",
    "build_tf_dataset",
    "dataset_from_pairs",
    "is_decodable",
    "load_image",
    "preprocess_arrays",
    "preprocess_image",
    "safe_load_image",
    "SplitResult",
    "build_class_names",
    "list_classes",
    "load_class_names",
    "make_splits",
    "save_class_names",
]
