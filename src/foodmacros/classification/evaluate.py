"""Evaluation metrics for the classifier.

Top-1 / top-5 accuracy and a full classification report + confusion matrix.
Class ``labels`` are always passed explicitly so the report works even when a
split does not contain every class, and the confusion-matrix diagonal is left
intact.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from ..data.preprocessing import dataset_from_pairs


def collect_predictions(
    model,
    pairs,
    img_size: int,
    batch_size: int,
    preprocess_fn=None,
    verbose: int = 0,
):
    """Return ``(y_true, probabilities)`` for ``(path, label)`` pairs."""
    pairs = list(pairs)
    if not pairs:
        raise ValueError("pairs is empty; nothing to evaluate.")
    ds = dataset_from_pairs(
        pairs, img_size, batch_size, shuffle=False, preprocess_fn=preprocess_fn
    )
    probs = np.asarray(model.predict(ds, verbose=verbose))
    y_true = np.asarray([int(label) for _, label in pairs], dtype=int)
    return y_true, probs


def topk_accuracy(y_true: np.ndarray, probs: np.ndarray, k: int = 5) -> float:
    if len(y_true) == 0:
        return float("nan")
    k = int(min(k, probs.shape[1]))
    top = np.argsort(-probs, axis=1)[:, :k]
    hits = [y_true[i] in top[i] for i in range(len(y_true))]
    return float(np.mean(hits))


def evaluate_predictions(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    if len(y_true) == 0:
        return {"accuracy": float("nan"), "top5_accuracy": float("nan"), "num_samples": 0}
    preds = np.argmax(probs, axis=1)
    return {
        "accuracy": float(np.mean(preds == y_true)),
        "top5_accuracy": topk_accuracy(y_true, probs, 5),
        "num_samples": int(len(y_true)),
    }


def classification_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: Sequence[str]) -> str:
    from sklearn.metrics import classification_report as skl_report

    labels = list(range(len(class_names)))
    return skl_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=list(class_names),
        zero_division=0,
    )


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: Sequence[str]) -> np.ndarray:
    from sklearn.metrics import confusion_matrix as skl_cm

    labels = list(range(len(class_names)))
    return skl_cm(y_true, y_pred, labels=labels)


def evaluate_model(
    model,
    pairs,
    class_names: Sequence[str],
    img_size: int,
    batch_size: int,
    preprocess_fn=None,
    verbose: int = 0,
) -> Dict[str, object]:
    """Full evaluation: metrics + per-class report + confusion matrix."""
    y_true, probs = collect_predictions(
        model, pairs, img_size, batch_size, preprocess_fn=preprocess_fn, verbose=verbose
    )
    y_pred = np.argmax(probs, axis=1)
    metrics: Dict[str, object] = evaluate_predictions(y_true, probs)
    metrics["report"] = classification_report(y_true, y_pred, class_names)
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, class_names)
    return metrics
