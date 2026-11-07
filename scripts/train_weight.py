#!/usr/bin/env python
"""Train RGB-only mass regressor on Nutrition5k (grams only).

Usage:
    python scripts/train_weight.py --nutrition5k /path/to/nutrition5k --output models/weight
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from foodmacros.weight.mass_dataset import build_weight_dataset, split_by_dish_id
from foodmacros.weight.regression import build_mass_only_regressor, get_regression_preprocess_input


def _compute_metrics(y_true, y_pred):
    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    # R2
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0:
        r2 = float("nan")
    else:
        r2 = 1.0 - ss_res / ss_tot
    # MAPE
    eps = 1e-8
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0)
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train RGB-only mass regressor on Nutrition5k")
    parser.add_argument("--nutrition5k", required=False, help="Path to Nutrition5k root")
    parser.add_argument("--output", default="models/weight", help="Output directory for model")
    parser.add_argument("--artifacts", default="artifacts/weight_dataset", help="Dataset splits dir")
    parser.add_argument("--eval-plots", default="artifacts/weight_evaluation", help="Evaluation plots dir")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fine-tune", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args(argv)

    np.random.seed(args.seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(args.seed)
    except Exception:
        pass

    nutrition5k_root = args.nutrition5k
    if nutrition5k_root is None:
        # try to discover
        try:
            from foodmacros.data.discovery import resolve_inputs
            from foodmacros.config import load_config

            cfg = load_config()
            resolved = resolve_inputs(cfg)
            nutrition5k_root = resolved.nutrition5k_root
        except Exception:
            nutrition5k_root = None

    if nutrition5k_root is None or not Path(nutrition5k_root).exists():
        print("Nutrition5k imagery not found.")
        print("Training was not executed.")
        print(f"Checked: {nutrition5k_root}")
        return 0

    print("Building Nutrition5k weight dataset (RGB only)...")
    artifacts_dir = Path(args.artifacts)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    df, stats = build_weight_dataset(nutrition5k_root, output_csv=artifacts_dir / "all_pairs.csv")
    print("Nutrition5k weight dataset")
    print("--------------------------")
    print(f"Metadata dishes: {stats['metadata_dishes']}")
    print(f"RGB images found: {stats['rgb_images_found']}")
    print(f"Valid image/weight pairs: {stats['valid_pairs']}")
    print(f"Missing images: {stats['missing_images']}")
    print(f"Missing weights: {stats['missing_weights']}")
    print(f"Invalid weights: {stats['invalid_weights']}")

    if len(df) == 0:
        print("No valid pairs found. Training was not executed.")
        return 0

    print("\nSplitting by dish_id (70/15/15)...")
    train_df, val_df, test_df = split_by_dish_id(df, 0.7, 0.15, 0.15, seed=args.seed)
    train_df.to_csv(artifacts_dir / "train.csv", index=False)
    val_df.to_csv(artifacts_dir / "validation.csv", index=False)
    test_df.to_csv(artifacts_dir / "test.csv", index=False)
    print(f"Train dishes/images: {train_df['dish_id'].nunique()}/{len(train_df)}")
    print(f"Val dishes/images: {val_df['dish_id'].nunique()}/{len(val_df)}")
    print(f"Test dishes/images: {test_df['dish_id'].nunique()}/{len(test_df)}")

    try:
        import tensorflow as tf
    except Exception as e:
        print(f"TensorFlow required for training: {e}")
        return 1

    # normalization from train only
    mean_mass = float(train_df["mass_g"].mean())
    std_mass = float(train_df["mass_g"].std()) if float(train_df["mass_g"].std()) > 0 else 1.0
    print(f"\nMass stats (train): mean={mean_mass:.2f} g, std={std_mass:.2f} g")

    preprocess_fn = get_regression_preprocess_input("mobilenetv2")

    def load_img(path):
        raw = tf.io.read_file(path)
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, (args.img_size, args.img_size))
        img = tf.cast(img, tf.float32)
        return img

    def make_ds(dataframe, shuffle=False):
        paths = dataframe["image_path"].tolist()
        masses = dataframe["mass_g"].astype(float).tolist()
        y = np.array([(m - mean_mass) / std_mass for m in masses], dtype=np.float32)
        ds = tf.data.Dataset.from_tensor_slices((paths, y))
        def _map(p, yy):
            img = load_img(p)
            return img, yy
        ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
        if shuffle:
            ds = ds.shuffle(min(len(paths), 10000), seed=args.seed)
        ds = ds.batch(args.batch_size)
        ds = ds.map(lambda x, yv: (preprocess_fn(x), yv), num_parallel_calls=tf.data.AUTOTUNE)
        return ds.prefetch(tf.data.AUTOTUNE)

    train_ds = make_ds(train_df, shuffle=True)
    val_ds = make_ds(val_df, shuffle=False)
    test_ds = make_ds(test_df, shuffle=False)

    model, base = build_mass_only_regressor(img_size=args.img_size, backbone="mobilenetv2", weights="imagenet")
    for layer in base.layers:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr), loss="mse", metrics=["mae"])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_mae", patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_mae", factor=0.5, patience=5, min_lr=1e-6),
    ]
    print("\nTraining head...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks, verbose=1)
    histories = {"head": history.history}

    if args.fine_tune:
        print("\nFine-tuning top layers...")
        for layer in base.layers[-30:]:
            layer.trainable = True
        model.compile(optimizer=tf.keras.optimizers.Adam(args.lr / 10), loss="mse", metrics=["mae"])
        fine_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=max(10, args.epochs // 5),
            callbacks=callbacks,
            verbose=1,
        )
        histories["fine_tune"] = fine_history.history

    eval_dir = Path(args.eval_plots)
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "history.json").write_text(json.dumps(histories, indent=2), encoding="utf-8")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 4))
        offset = 0
        for phase, hist in histories.items():
            epochs_range = range(offset + 1, offset + len(hist["loss"]) + 1)
            plt.subplot(1, 2, 1)
            plt.plot(epochs_range, hist["loss"], label=f"{phase} train loss")
            plt.plot(epochs_range, hist.get("val_loss", []), label=f"{phase} val loss")
            plt.subplot(1, 2, 2)
            plt.plot(epochs_range, hist["mae"], label=f"{phase} train MAE")
            plt.plot(epochs_range, hist.get("val_mae", []), label=f"{phase} val MAE")
            offset += len(hist["loss"])
        plt.subplot(1, 2, 1)
        plt.xlabel("Epoch")
        plt.ylabel("MSE loss")
        plt.legend()
        plt.title("Training / validation loss")
        plt.subplot(1, 2, 2)
        plt.xlabel("Epoch")
        plt.ylabel("MAE (normalized)")
        plt.legend()
        plt.title("Training / validation MAE")
        plt.tight_layout()
        plt.savefig(eval_dir / "training_curves.png")
        plt.close()
        print(f"\nSaved training history to {eval_dir / 'history.json'}")
        print(f"Saved training curves to {eval_dir / 'training_curves.png'}")
    except Exception as exc:
        print(f"Training curve plotting skipped: {exc}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "mass_regressor.keras"
    model.save(model_path)
    norm = {"mode": "mass_only", "target": "mass_g", "mean": mean_mass, "std": std_mass, "img_size": args.img_size, "backbone": "mobilenetv2"}
    (out_dir / "normalization.json").write_text(json.dumps(norm, indent=2))
    cfg_out = {"img_size": args.img_size, "backbone": "mobilenetv2", "preprocessing": "mobilenet_v2.preprocess_input"}
    (out_dir / "config.json").write_text(json.dumps(cfg_out, indent=2))
    print(f"\nSaved model to {model_path}")

    # baseline
    baseline_pred = np.array([mean_mass] * len(test_df))
    y_test = test_df["mass_g"].values.astype(float)
    baseline_metrics = _compute_metrics(y_test, baseline_pred)
    print("\nBaseline (mean train mass):")
    print(f"  MAE: {baseline_metrics['mae']:.2f} g, RMSE: {baseline_metrics['rmse']:.2f} g, R2: {baseline_metrics['r2']:.3f}, MAPE: {baseline_metrics['mape']:.2f}%")

    # predict test
    preds_norm = model.predict(test_ds, verbose=0)
    preds = preds_norm.reshape(-1) * std_mass + mean_mass
    metrics = _compute_metrics(y_test, preds)
    print("\nTest set (RGB-only):")
    print(f"  MAE: {metrics['mae']:.2f} g, RMSE: {metrics['rmse']:.2f} g, R2: {metrics['r2']:.3f}, MAPE: {metrics['mape']:.2f}%")
    print(f"  Mean actual: {y_test.mean():.2f} g, Median actual: {np.median(y_test):.2f} g")
    print(f"  Mean pred: {preds.mean():.2f} g, Median pred: {np.median(preds):.2f} g")

    # ranges
    ranges = [(0, 100), (100, 250), (250, 500), (500, 1000), (1000, 10**12)]
    print("\nMAE by weight range:")
    for lo, hi in ranges:
        mask = (y_test >= lo) & (y_test < hi)
        if mask.sum() == 0:
            continue
        mae_r = float(np.mean(np.abs(y_test[mask] - preds[mask])))
        print(f"  {lo}-{hi if hi < 10**12 else '+'} g (n={mask.sum()}): {mae_r:.2f} g")

    test_metrics = {
        "baseline_mae": baseline_metrics["mae"],
        "test_mae": metrics["mae"],
        "test_rmse": metrics["rmse"],
        "test_r2": metrics["r2"],
        "test_mape": metrics["mape"],
        "mean_actual": float(y_test.mean()),
        "median_actual": float(np.median(y_test)),
        "mean_pred": float(preds.mean()),
        "median_pred": float(np.median(preds)),
        "n_test": int(len(y_test)),
    }
    (out_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))

    # plots
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_dir = Path(args.eval_plots)
        plot_dir.mkdir(parents=True, exist_ok=True)
        # 1 actual vs pred
        plt.figure()
        plt.scatter(y_test, preds, alpha=0.5, s=10)
        lims = [0, max(y_test.max(), preds.max()) * 1.05]
        plt.plot(lims, lims, "r--", label="y=x")
        plt.xlabel("Actual mass (g)")
        plt.ylabel("Predicted mass (g)")
        plt.title("Actual vs Predicted")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "actual_vs_predicted.png")
        plt.close()
        # 2 error distribution
        errors = preds - y_test
        plt.figure()
        plt.hist(errors, bins=50)
        plt.xlabel("Predicted - Actual (g)")
        plt.ylabel("Count")
        plt.title("Prediction error distribution")
        plt.tight_layout()
        plt.savefig(plot_dir / "error_distribution.png")
        plt.close()
        # 3 mae by range
        range_names = []
        range_maes = []
        for lo, hi in ranges:
            mask = (y_test >= lo) & (y_test < hi)
            if mask.sum() == 0:
                continue
            mae_r = float(np.mean(np.abs(y_test[mask] - preds[mask])))
            range_names.append(f"{lo}-{hi if hi < 10**12 else '1000+'}")
            range_maes.append(mae_r)
        plt.figure()
        plt.bar(range_names, range_maes)
        plt.ylabel("MAE (g)")
        plt.title("MAE by weight range")
        plt.tight_layout()
        plt.savefig(plot_dir / "mae_by_range.png")
        plt.close()
        print(f"\nSaved plots to {plot_dir}")
    except Exception as e:
        print(f"Plotting skipped: {e}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
