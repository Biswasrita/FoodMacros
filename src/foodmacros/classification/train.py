"""Two-stage classifier training (frozen head, then partial fine-tune).

Everything honours ``config`` (quick vs full, learning rates, epochs, batch
size) and writes portable artifacts (`.keras`, class names, label mapping).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..config import AppConfig
from ..data.preprocessing import dataset_from_pairs
from ..data.splits import save_class_names
from ..env_checks import default_working_dir
from .model import (
    BuiltModel,
    build_model,
    compile_model,
    get_preprocess_input,
    set_backbone_trainable,
    unfreeze_backbone_fraction,
)

Pair = Tuple[str, int]


@dataclass
class TrainResult:
    model_path: str
    class_names_path: str
    label_mapping_path: str
    output_dir: str
    quick: bool
    epochs_head: int
    epochs_finetune: int
    history_head: Dict[str, list] = field(default_factory=dict)
    history_finetune: Dict[str, list] = field(default_factory=dict)
    built: Any = None

    def summary(self) -> str:
        return (
            f"output_dir={self.output_dir} quick={self.quick} "
            f"epochs=head:{self.epochs_head},finetune:{self.epochs_finetune} "
            f"model={self.model_path}"
        )


def _resolve_output_dir(config: AppConfig, output_dir) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    if config.runtime.output_dir:
        return Path(config.runtime.output_dir)
    return default_working_dir()


def save_artifacts(
    model, class_names: Sequence[str], output_dir, config: Optional[AppConfig] = None
) -> tuple[str, str, str]:
    """Save model + class index + label mapping (+ config) to ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "classifier.keras"
    model.save(model_path)
    names_path = save_class_names(output_dir / "class_names.json", class_names)

    mapping = {str(i): str(name) for i, name in enumerate(class_names)}
    mapping_path = output_dir / "label_mapping.json"
    mapping_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")

    if config is not None:
        config.save(output_dir / "config_used.json")

    return str(model_path), str(names_path), str(mapping_path)


def train_classifier(
    config: AppConfig,
    class_names: Sequence[str],
    train_pairs: Sequence[Pair],
    val_pairs: Sequence[Pair],
    output_dir=None,
    built: Optional[BuiltModel] = None,
    weights: str | None = "imagenet",
    verbose: int = 2,
) -> TrainResult:
    """Train the classifier on ``(path, label)`` pairs.

    ``weights=None`` is supported (used for fast smoke tests). ``built`` lets
    callers inject a pre-built model (e.g. deterministic test models).
    """
    import tensorflow as tf

    if not train_pairs:
        raise ValueError("train_pairs is empty; nothing to train on.")
    if not val_pairs:
        raise ValueError("val_pairs is empty; a validation set is required for checkpoints.")

    quick = config.runtime.mode == "quick"
    out = _resolve_output_dir(config, output_dir)
    out.mkdir(parents=True, exist_ok=True)

    num_classes = len(class_names)
    if built is None:
        built = build_model(
            num_classes=num_classes,
            img_size=config.model.img_size,
            architecture=config.model.architecture,
            dropout=config.model.dropout,
            use_vit_fusion=config.model.use_vit_fusion,
            vit_preset=config.model.vit_preset,
            weights=weights,
        )

    preprocess_fn = get_preprocess_input(config.model.architecture)
    batch_size = config.training.batch_size
    train_ds = dataset_from_pairs(
        train_pairs, config.model.img_size, batch_size,
        shuffle=True, seed=config.splits.shuffle_seed, preprocess_fn=preprocess_fn,
    )
    val_ds = dataset_from_pairs(
        val_pairs, config.model.img_size, batch_size,
        shuffle=False, preprocess_fn=preprocess_fn,
    )

    steps = config.training.quick_steps_per_epoch if quick else None
    epochs_head = config.training.quick_epochs if quick else config.training.epochs_head
    epochs_finetune = config.training.quick_epochs if quick else config.training.epochs_finetune

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(out / "best.keras"), monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=config.training.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(1, config.training.early_stopping_patience // 2),
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(out / "training_log.csv")),
    ]

    # Stage A: train the head with a frozen backbone.
    set_backbone_trainable(built, False)
    compile_model(built, config.training.learning_rate_head, config.training.label_smoothing)
    hist_head = built.model.fit(
        train_ds, validation_data=val_ds, epochs=epochs_head,
        steps_per_epoch=steps, callbacks=callbacks, verbose=verbose,
    )

    # Stage B: unfreeze the top of the backbone and fine-tune at a lower LR.
    unfreeze_backbone_fraction(built, fraction=0.1)
    compile_model(built, config.training.learning_rate_finetune, config.training.label_smoothing)
    hist_finetune = built.model.fit(
        train_ds, validation_data=val_ds, epochs=epochs_finetune,
        steps_per_epoch=steps, callbacks=callbacks, verbose=verbose,
    )

    model_path, names_path, mapping_path = save_artifacts(
        built.model, class_names, out, config
    )
    return TrainResult(
        model_path=model_path,
        class_names_path=names_path,
        label_mapping_path=mapping_path,
        output_dir=str(out),
        quick=quick,
        epochs_head=epochs_head,
        epochs_finetune=epochs_finetune,
        history_head=dict(hist_head.history),
        history_finetune=dict(hist_finetune.history),
        built=built,
    )
