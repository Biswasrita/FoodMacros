"""Image decoding, resizing and tf.data construction.

Model-agnostic: images are decoded to RGB ``uint8`` and optionally converted
to ``float32``. Any model-specific normalisation (e.g. EfficientNet's
``preprocess_input``) is passed in as a callable by the caller.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"}


class ImageDecodeError(Exception):
    """Raised when an image cannot be opened or decoded."""


def is_decodable(path: str | Path) -> bool:
    """Return True if PIL can open and verify the file as an image."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def load_image(path: str | Path, img_size: int) -> np.ndarray:
    """Load an image as an RGB ``uint8`` array of shape ``(img_size, img_size, 3)``.

    Raises :class:`ImageDecodeError` with the offending path on failure so
    callers can report precisely which file was invalid.
    """
    from PIL import Image

    try:
        with Image.open(path) as im:
            im = im.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
            return np.asarray(im, dtype=np.uint8)
    except Exception as exc:
        raise ImageDecodeError(f"Could not decode image '{path}': {exc}") from exc


def safe_load_image(path: str | Path, img_size: int) -> np.ndarray | None:
    """Like :func:`load_image` but returns ``None`` instead of raising."""
    try:
        return load_image(path, img_size)
    except ImageDecodeError:
        return None


def preprocess_image(
    path: str | Path,
    img_size: int,
    preprocess_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Return a batched ``float32`` array ``(1, img_size, img_size, 3)``.

    ``preprocess_fn`` (if given) is applied to the ``H, W, 3`` array before
    batching, e.g. ``tensorflow.keras.applications.efficientnet.preprocess_input``.
    """
    array = load_image(path, img_size)
    if preprocess_fn is not None:
        array = preprocess_fn(array.astype(np.float32))
    else:
        array = array.astype(np.float32)
    return np.expand_dims(array, axis=0)


def preprocess_arrays(
    arrays: Iterable[np.ndarray],
    preprocess_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Stack RGB uint8 arrays into a batched float32 array."""
    batch = np.stack([np.asarray(a, dtype=np.uint8) for a in arrays], axis=0)
    batch = batch.astype(np.float32)
    if preprocess_fn is not None:
        batch = preprocess_fn(batch)
    return batch


def build_tf_dataset(
    root: str | Path,
    class_names: Sequence[str],
    img_size: int,
    batch_size: int,
    shuffle: bool = False,
    seed: int | None = None,
    subset: str | None = None,
    validation_split: float | None = None,
):
    """Create a ``tf.data.Dataset`` with a guaranteed, explicit class order.

    The directory's subfolders must match ``class_names`` exactly; otherwise a
    clear error is raised to prevent silent label-index drift.
    """
    import tensorflow as tf

    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Image root does not exist: {root}")

    present = sorted(
        p.name for p in root.iterdir() if p.is_dir() and any(
            f.suffix.lower() in IMAGE_EXTS for f in p.iterdir() if f.is_file()
        )
    )
    expected = sorted(class_names)
    if present != expected:
        missing = sorted(set(expected) - set(present))
        unexpected = sorted(set(present) - set(expected))
        raise ValueError(
            f"Class folders under {root} do not match class_names. "
            f"Missing: {missing}. Unexpected: {unexpected}."
        )

    kwargs = dict(
        directory=str(root),
        labels="inferred",
        label_mode="int",
        class_names=list(class_names),
        color_mode="rgb",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )
    if validation_split is not None:
        kwargs["validation_split"] = validation_split
        kwargs["subset"] = subset
    return tf.keras.utils.image_dataset_from_directory(**kwargs)


def dataset_from_pairs(
    pairs,
    img_size: int,
    batch_size: int,
    shuffle: bool = False,
    seed: int | None = None,
    preprocess_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    cache: bool = False,
):
    """Build a streaming ``tf.data.Dataset`` from ``(path, label)`` pairs.

    Images are decoded, resized to ``(img_size, img_size)`` and returned as
    ``float32`` in [0, 255]; ``preprocess_fn`` (if given) is applied after
    batching. This streams from disk, so it scales to full Food-101.
    """
    import tensorflow as tf

    pairs = list(pairs)
    paths = [str(p) for p, _ in pairs]
    labels = [int(l) for _, l in pairs]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _load(path, label):
        raw = tf.io.read_file(path)
        image = tf.io.decode_image(raw, channels=3, expand_animations=False)
        image = tf.image.resize(image, (img_size, img_size))
        return tf.cast(image, tf.float32), label

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(paths), 10000), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    if preprocess_fn is not None:
        ds = ds.map(lambda x, y: (preprocess_fn(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)
