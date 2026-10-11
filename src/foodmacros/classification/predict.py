"""Single-image prediction and classifier loading.

Prediction is robust: invalid inputs return an *unavailable* result with a
reason instead of raising, so the end-to-end pipeline/UI can degrade cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..data.preprocessing import ImageDecodeError, load_image


@dataclass
class PredictionResult:
    available: bool
    top_k: List[Tuple[str, float]] = field(default_factory=list)
    label: Optional[str] = None
    confidence: Optional[float] = None
    reason: str = ""

    def summary(self) -> str:
        if not self.available:
            return f"unavailable: {self.reason}"
        top = ", ".join(f"{name} ({prob:.1%})" for name, prob in self.top_k)
        return f"{self.label} ({self.confidence:.1%}) | top-{len(self.top_k)}: {top}"


def load_classifier(model_dir):
    """Reconstruct a saved classifier for inference.

    Loads with ``compile=False`` so a training-only custom loss does not need
    to be importable at inference time.
    """
    import tensorflow as tf

    from ..data.splits import load_class_names

    model_dir = Path(model_dir)
    model_path = model_dir / "classifier.keras"
    names_path = model_dir / "class_names.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model not found: {model_path}")
    if not names_path.exists():
        raise FileNotFoundError(f"Class names file not found: {names_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    class_names = load_class_names(names_path)
    return model, class_names


def _to_model_input(array, img_size: int) -> np.ndarray:
    """Coerce an RGB array to ``float32`` at the model's input size."""
    arr = np.asarray(array)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected an HxWx3 RGB array, got shape {arr.shape}.")
    if arr.shape[0] != img_size or arr.shape[1] != img_size:
        import tensorflow as tf

        arr = tf.image.resize(arr, (img_size, img_size)).numpy()
    return arr.astype(np.float32)


def predict_array(
    model,
    preprocess_fn,
    array,
    class_names: Sequence[str],
    img_size: int,
    top_k: int = 3,
) -> PredictionResult:
    if array is None:
        return PredictionResult(False, reason="no image provided")
    try:
        x = _to_model_input(array, img_size)
    except ValueError as exc:
        return PredictionResult(False, reason=str(exc))
    if preprocess_fn is not None:
        x = preprocess_fn(x)
    probs = np.asarray(model.predict(x[None, ...], verbose=0))[0]
    k = int(min(max(top_k, 1), len(class_names)))
    order = np.argsort(-probs)[:k]
    top = [(str(class_names[int(i)]), float(probs[int(i)])) for i in order]
    return PredictionResult(True, top_k=top, label=top[0][0], confidence=top[0][1])


def predict_path(
    model,
    preprocess_fn,
    path,
    class_names: Sequence[str],
    img_size: int,
    top_k: int = 3,
) -> PredictionResult:
    try:
        array = load_image(path, img_size)
    except ImageDecodeError as exc:
        return PredictionResult(False, reason=str(exc))
    return predict_array(model, preprocess_fn, array, class_names, img_size, top_k=top_k)
