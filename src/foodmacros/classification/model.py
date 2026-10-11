"""Classification model construction.

Default backbone is EfficientNetB0. An optional, config-gated ViT fusion
concatenates pooled EfficientNet features with pooled ViT features. The ViT
branch requires the optional ``keras-hub`` package and is a no-op (with a
clear error) when the flag is on but the dependency is missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

# Supported ImageNet backbones. Only the ones listed are built by name.
SUPPORTED_BACKBONES = ("efficientnetb0", "mobilenetv2", "resnet50")


@dataclass
class BuiltModel:
    model: Any
    backbones: Dict[str, Any]
    architecture: str
    img_size: int
    num_classes: int
    use_vit_fusion: bool = False

    def backbone_layers(self):
        return self.backbones


def get_preprocess_input(architecture: str = "efficientnetb0"):
    """Return the Keras ``preprocess_input`` for an ImageNet backbone."""
    import tensorflow as tf

    name = architecture.lower()
    if name.startswith("efficientnet"):
        return tf.keras.applications.efficientnet.preprocess_input
    if name.startswith("mobilenet"):
        return tf.keras.applications.mobilenet_v2.preprocess_input
    if name.startswith("resnet"):
        return tf.keras.applications.resnet50.preprocess_input
    # Fallback: identity (model handles its own scaling).
    return lambda x: x


def build_backbone(architecture: str, img_size: int, weights: str | None = "imagenet"):
    """Build a convolutional backbone with ``include_top=False``."""
    import tensorflow as tf

    name = architecture.lower()
    apps = tf.keras.applications
    kwargs = dict(include_top=False, weights=weights, input_shape=(img_size, img_size, 3))
    if name == "efficientnetb0":
        return apps.EfficientNetB0(**kwargs)
    if name == "mobilenetv2":
        return apps.MobileNetV2(**kwargs)
    if name == "resnet50":
        return apps.ResNet50(**kwargs)
    raise ValueError(
        f"Unsupported architecture '{architecture}'. Supported: {list(SUPPORTED_BACKBONES)}."
    )


def _build_vit_backbone(vit_preset: str, img_size: int):
    """Build a ViT backbone from KerasHub (optional dependency).

    Raises a clear, actionable error when ``keras-hub`` is unavailable or the
    preset cannot be loaded, rather than crashing obscurely.
    """
    try:
        import keras_hub
    except ImportError as exc:
        raise RuntimeError(
            "ViT fusion requires the optional 'keras-hub' package. Install it "
            "(pip install keras-hub) and ensure pretrained weights are reachable, "
            "or set model.use_vit_fusion=false."
        ) from exc
    try:
        return keras_hub.models.ViTBackbone.from_preset(
            vit_preset, image_shape=(img_size, img_size, 3)
        )
    except Exception as exc:  # preset/API mismatch must be reported clearly
        raise RuntimeError(
            f"Could not load ViT preset '{vit_preset}' from KerasHub: "
            f"{type(exc).__name__}: {exc}. Check the preset name / installed "
            "keras-hub version, or disable model.use_vit_fusion."
        ) from exc


def build_model(
    num_classes: int,
    img_size: int = 224,
    architecture: str = "efficientnetb0",
    dropout: float = 0.3,
    use_vit_fusion: bool = False,
    vit_preset: str = "vit_base_patch16_224",
    weights: str | None = "imagenet",
) -> BuiltModel:
    """Build the classifier (EfficientNet + optional ViT fusion + dense head)."""
    import tensorflow as tf

    if num_classes < 2:
        raise ValueError("num_classes must be >= 2.")

    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="image")
    cnn = build_backbone(architecture, img_size, weights=weights)
    cnn.trainable = False
    cnn_features = tf.keras.layers.GlobalAveragePooling2D(name="cnn_gap")(cnn(inputs))
    backbones: Dict[str, Any] = {architecture: cnn}

    if use_vit_fusion:
        vit = _build_vit_backbone(vit_preset, img_size)
        vit.trainable = False
        vit_out = vit(inputs)
        if len(vit_out.shape) == 4:
            vit_features = tf.keras.layers.GlobalAveragePooling2D(name="vit_gap")(vit_out)
        else:
            vit_features = tf.keras.layers.LayerNormalization(name="vit_norm")(vit_out)
        features = tf.keras.layers.Concatenate(name="fusion")([cnn_features, vit_features])
        backbones["vit"] = vit
    else:
        features = cnn_features

    x = tf.keras.layers.Dropout(dropout, name="head_dropout")(features)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    model = tf.keras.Model(inputs, outputs, name=f"{architecture}_classifier")
    return BuiltModel(
        model=model,
        backbones=backbones,
        architecture=architecture,
        img_size=img_size,
        num_classes=num_classes,
        use_vit_fusion=use_vit_fusion,
    )


def make_loss(label_smoothing: float = 0.0):
    """Return a sparse softmax loss, with label smoothing if requested.

    Keras 3's ``SparseCategoricalCrossentropy`` no longer accepts
    ``label_smoothing``, so smoothing is implemented explicitly here.
    """
    import tensorflow as tf

    if label_smoothing <= 0:
        return tf.keras.losses.SparseCategoricalCrossentropy(name="sparse_cce")

    class SparseCategoricalCrossentropySmoothed(tf.keras.losses.Loss):
        def __init__(self):
            super().__init__(name="sparse_cce_smoothed")

        def call(self, y_true, y_pred):
            y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
            classes = tf.shape(y_pred)[-1]
            one_hot = tf.one_hot(y_true, classes)
            n = tf.cast(classes, y_pred.dtype)
            s = tf.cast(label_smoothing, y_pred.dtype)
            smooth = (1.0 - s) * one_hot + s / n
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
            return -tf.reduce_sum(smooth * tf.math.log(y_pred), axis=-1)

    return SparseCategoricalCrossentropySmoothed()


def compile_model(built: BuiltModel, learning_rate: float = 1e-3, label_smoothing: float = 0.0):
    import tensorflow as tf

    built.model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=make_loss(label_smoothing),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5"),
        ],
    )
    return built


def set_backbone_trainable(built: BuiltModel, trainable: bool) -> None:
    for backbone in built.backbones.values():
        backbone.trainable = trainable


def unfreeze_backbone_fraction(
    built: BuiltModel, fraction: float = 0.1, freeze_batchnorm: bool = True
) -> int:
    """Unfreeze the last ``fraction`` of each backbone's layers.

    BatchNormalization layers stay frozen by default (safer with small
    batches). Returns the number of layers made trainable.
    """
    import tensorflow as tf

    if not (0.0 <= fraction <= 1.0):
        raise ValueError("fraction must be in [0, 1].")
    unfrozen = 0
    for backbone in built.backbones.values():
        layers = backbone.layers
        count = int(round(len(layers) * fraction))
        for layer in layers[len(layers) - count:] if count else []:
            if freeze_batchnorm and isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True
                unfrozen += 1
        backbone.trainable = True
    return unfrozen
