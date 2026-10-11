"""Stage 4 tests for evaluation metrics (pure numpy/sklearn; no TF needed)."""

import numpy as np

from foodmacros.classification.evaluate import (
    classification_report,
    confusion_matrix,
    evaluate_predictions,
    topk_accuracy,
)


def test_evaluate_predictions_perfect():
    y_true = np.array([0, 1, 2])
    probs = np.eye(3)
    metrics = evaluate_predictions(y_true, probs)
    assert metrics["accuracy"] == 1.0
    assert metrics["top5_accuracy"] == 1.0
    assert metrics["num_samples"] == 3


def test_evaluate_predictions_empty_is_nan():
    metrics = evaluate_predictions(np.array([]), np.zeros((0, 3)))
    assert metrics["num_samples"] == 0


def test_topk_accuracy_catches_lower_rank():
    y_true = np.array([2])
    probs = np.array([[0.4, 0.3, 0.2, 0.05, 0.05]])
    assert topk_accuracy(y_true, probs, 5) == 1.0
    assert topk_accuracy(y_true, probs, 2) == 0.0


def test_classification_report_handles_absent_class():
    y_true = np.array([0, 0])
    y_pred = np.array([0, 1])
    report = classification_report(y_true, y_pred, ["a", "b", "c"])
    assert "a" in report and "b" in report and "c" in report


def test_confusion_matrix_diagonal_is_intact():
    y_true = np.array([0, 0, 1])
    y_pred = np.array([0, 0, 1])
    cm = confusion_matrix(y_true, y_pred, ["a", "b"])
    assert cm[0, 0] == 2
    assert cm[1, 1] == 1
    assert cm[1, 0] == 0
