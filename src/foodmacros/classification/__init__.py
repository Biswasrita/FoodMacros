"""Classification: model building, training, evaluation and prediction."""

from .evaluate import (
    classification_report,
    collect_predictions,
    confusion_matrix,
    evaluate_model,
    evaluate_predictions,
    topk_accuracy,
)
from .model import (
    SUPPORTED_BACKBONES,
    BuiltModel,
    build_backbone,
    build_model,
    compile_model,
    get_preprocess_input,
    make_loss,
    set_backbone_trainable,
    unfreeze_backbone_fraction,
)
from .predict import PredictionResult, load_classifier, predict_array, predict_path
from .train import TrainResult, save_artifacts, train_classifier

__all__ = [
    "SUPPORTED_BACKBONES",
    "BuiltModel",
    "build_backbone",
    "build_model",
    "compile_model",
    "get_preprocess_input",
    "make_loss",
    "set_backbone_trainable",
    "unfreeze_backbone_fraction",
    "classification_report",
    "collect_predictions",
    "confusion_matrix",
    "evaluate_model",
    "evaluate_predictions",
    "topk_accuracy",
    "PredictionResult",
    "load_classifier",
    "predict_array",
    "predict_path",
    "TrainResult",
    "save_artifacts",
    "train_classifier",
]
