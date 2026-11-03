"""Single-image prediction and classifier loading.

Prediction is robust: invalid inputs return an *unavailable* result with a
reason instead of raising, so the end-to-end pipeline/UI can degrade cleanly.
"""

from __future__ import annotations

import json
import logging
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..data.preprocessing import ImageDecodeError, load_image

logger = logging.getLogger(__name__)

# Layer-config keys written by newer Keras releases (>= 3.13) that older
# releases (e.g. 3.12) reject outright. They are optional tuning hints, so a
# ``None`` value carries no information and can be dropped safely.
_NEWER_KERAS_ONLY_KEYS = ("quantization_config",)


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


def _strip_null_newer_keras_keys(node) -> int:
    """Recursively drop ``None`` values for keys newer Keras added.

    Only ``<key>: None`` is removed (a non-null value is meaningful and left
    untouched so we never silently ignore real quantisation settings).
    Returns the number of removed entries.
    """
    removed = 0
    if isinstance(node, dict):
        for key in _NEWER_KERAS_ONLY_KEYS:
            if key in node and node[key] is None:
                del node[key]
                removed += 1
        for value in node.values():
            removed += _strip_null_newer_keras_keys(value)
    elif isinstance(node, list):
        for value in node:
            removed += _strip_null_newer_keras_keys(value)
    return removed


def _load_model_with_legacy_config(model_path: Path):
    """Load a ``.keras`` archive saved by a newer Keras than the installed one.

    Newer Keras serialises ``quantization_config: None`` into layer configs.
    The installed Keras has no such argument and refuses to deserialize the
    layer, even though the saved weights do not depend on it. We rewrite only
    the archive's ``config.json`` (in a temporary copy, never touching the
    original) dropping those null entries, then load it normally.
    """
    import tensorflow as tf

    with tempfile.TemporaryDirectory(prefix="foodmacros-classifier-") as tmp:
        patched = Path(tmp) / model_path.name
        removed = 0
        with zipfile.ZipFile(model_path) as src, zipfile.ZipFile(
            patched, "w", zipfile.ZIP_DEFLATED
        ) as dst:
            for info in src.infolist():
                data = src.read(info.filename)
                if info.filename == "config.json":
                    config = json.loads(data)
                    removed = _strip_null_newer_keras_keys(config)
                    data = json.dumps(config).encode("utf-8")
                dst.writestr(info, data)
        if not removed:
            raise ValueError(
                f"{model_path} could not be loaded and no compatible null "
                "configuration entries were found to strip."
            )
        logger.warning(
            "Loaded %s through the legacy-config compatibility path "
            "(stripped %d null 'quantization_config' entr%s). Consider "
            "re-saving it with the installed Keras version.",
            model_path,
            removed,
            "y" if removed == 1 else "ies",
        )
        return tf.keras.models.load_model(patched, compile=False)


def _load_keras_model(model_path: Path):
    """Load a saved Keras model, tolerating newer-Keras serialization fields.

    Native loading is always tried first. The compatibility path is only used
    for the specific "unrecognized ``quantization_config``" failure and only
    strips null values; ``safe_mode=False`` is deliberately never used.
    """
    import tensorflow as tf

    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except (TypeError, ValueError) as exc:
        if not any(key in str(exc) for key in _NEWER_KERAS_ONLY_KEYS):
            raise
        try:
            return _load_model_with_legacy_config(model_path)
        except Exception as compat_exc:
            raise exc from compat_exc


def load_classifier(model_dir):
    """Reconstruct a saved classifier for inference.

    Loads with ``compile=False`` so a training-only custom loss does not need
    to be importable at inference time.
    """
    from ..data.splits import load_class_names

    model_dir = Path(model_dir)
    model_path = model_dir / "classifier.keras"
    names_path = model_dir / "class_names.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model not found: {model_path}")
    if not names_path.exists():
        raise FileNotFoundError(f"Class names file not found: {names_path}")
    model = _load_keras_model(model_path)
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
