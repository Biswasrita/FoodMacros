"""FoodMacros: food-image classification, optional detection, portion-weight
estimation and macronutrient calculation.
"""

from .config import (
    AppConfig,
    DEFAULT_CONFIG_PATH,
    PROJECT_ROOT,
    config_from_dict,
    load_config,
)
from .env_checks import (
    configure_mixed_precision,
    default_working_dir,
    format_report,
    get_env_report,
    has_package,
)
from .pipeline import AnalysisResult, FoodAnalysisPipeline, NutritionOutcome
from .multi_pipeline import DishResult, MultiAnalysisResult, MultiDishPipeline
from .multi_reporting import format_multi_analysis, multi_analysis_to_dict
from .reporting import LIMITATIONS, analysis_to_dict, format_analysis, format_nutrients

__all__ = [
    "AppConfig",
    "DEFAULT_CONFIG_PATH",
    "PROJECT_ROOT",
    "config_from_dict",
    "load_config",
    "configure_mixed_precision",
    "default_working_dir",
    "format_report",
    "get_env_report",
    "has_package",
    "AnalysisResult",
    "FoodAnalysisPipeline",
    "NutritionOutcome",
    "DishResult",
    "MultiAnalysisResult",
    "MultiDishPipeline",
    "LIMITATIONS",
    "analysis_to_dict",
    "format_analysis",
    "format_nutrients",
    "format_multi_analysis",
    "multi_analysis_to_dict",
]

__version__ = "0.1.0"
