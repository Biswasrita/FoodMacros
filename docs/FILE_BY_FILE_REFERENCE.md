# File-by-File Reference

This document lists **every file** in the repository, one line each, grouped by
directory. Sizes are from the audit at `CLEANUP_AUDIT.md` (snapshot of the
project before this documentation task). New files added by the documentation
task are listed at the end.

Legend: **R** = run/must-have, **G** = generated (safe to regenerate/delete),
**C** = config, **D** = documentation, **A** = trained artifact (do not modify).

---

## Project root (7 files)

| File | Size | Role |
|---|---|---|
| `README.md` | 4.6 KB | **D** Project front door; rewritten by this documentation task (old copy kept at `backups/README.md`) |
| `requirements.txt` | 848 B | **C** Core packages: numpy, pandas, scikit-learn, tensorflow, Pillow (minimum versions); optional extras listed as comments |
| `requirements-dev.txt` | 146 B | **C** Test/dev packages: pytest, nbformat |
| `.gitignore` | 249 B | **C** Ignores `outputs/`, `*.keras`, `*.h5`, `*.pt`, `*.ckpt`, `__pycache__/`, caches, venvs, notebooks clutter, and local `data/` |
| `nutrition.csv` | 28 KB | **A** Copy of `data/nutrition.csv` (byte-identical). The runtime uses `data/nutrition.csv`; this root copy is redundant |
| `yolov8n.pt` | 6.6 MB | **A** Copy of `models/food_detector/food_detector.pt` (byte-identical). Ultralytics food detector, 113 classes, **not** COCO |
| `Food_101_Nutrition_Fixed.ipynb` | 33 KB | **D** An **unexecuted** editor copy of `backups/Food_101_Nutrition_Fixed.backup.ipynb` (byte-identical, 46 cells, no outputs). The real training notebook is `model_outputs/notebook82076030c6 (1).ipynb`. See `NOTEBOOK_GUIDE.md` |

## `app/` (2 files)

| File | Size | Role |
|---|---|---|
| `gradio_app.py` | 12.8 KB | **R** The web UI (imports Gradio lazily; reports clearly if missing). Flags: `--config`, `--model-dir`, `--regressor-dir`, `--share`, `--port`. Single-dish tab + Multiple-dishes tab (two-step flow) |
| `__pycache__/` (1 .pyc) | G | Compiled bytecode, auto-generated |

## `backups/` (2 files)

| File | Size | Role |
|---|---|---|
| `Food_101_Nutrition_Fixed.backup.ipynb` | 33 KB | **D** Backup of the unexecuted editor notebook |
| `README.md` | 4.6 KB | **D** The original README, saved before this documentation task rewrote `README.md` |

## `configs/` (3 files)

| File | Size | Role |
|---|---|---|
| `default.json` | 1.5 KB | **C** Main configuration: runtime, data, splits, model, training, nutrition, weight, detection, ui |
| `multi_dish.json` | 213 B | **C** Multi-dish/detection overrides: food detector enabled, `conf 0.25`, `iou 0.45`, `verify_class_names=false`, `max_dishes 5`, `crop_padding 0.08` |
| `food_densities.json` | 1.3 KB | **C** Approximate densities (g/ml) used only by the experimental depth path (disabled by default) |

## `data/` (1 file)

| File | Size | Role |
|---|---|---|
| `nutrition.csv` | 28 KB | **A** The nutrition table: 505 rows = 101 Food-101 labels × 5 weight rows each (80/100/120/150/200 g per label mapping). Columns: `label, weight, calories, protein, carbohydrates, fats, fiber, sugars, sodium`. Runtime artifact - do not delete |

## `docs/` (examples in this project)

| File | Role |
|---|---|
| `architecture.md` | **D** Original technical architecture notes |
| `limitations.md` | **D** Original honesty notes (unverified pieces, weight semantics, detection notes) |
| `multi_dish.md` | **D** Original multi-dish feature guide |
| `PROJECT_GUIDE.md` | **D** (new) Conceptual guide to the whole project |
| `FILE_BY_FILE_REFERENCE.md` | **D** (new) This file |
| `MODELS_DATASETS_AND_METRICS.md` | **D** (new) Datasets, architectures, real accuracy numbers |
| `NOTEBOOK_GUIDE.md` | **D** (new) The three notebooks, cell by cell |
| `RUN_AND_TROUBLESHOOT.md` | **D** (new) Commands + troubleshooting |
| `CLEANUP_AUDIT.md` | **D** (new) Cleanup proposal (nothing deleted without approval) |
| `FRIEND_HANDOVER.md` | **D** (new) Submission-ready summary |

## `models/` (3 files, ~59 MB) - the runtime artifacts

| File | Size | Role |
|---|---|---|
| `classifier/classifier.keras` | 52.5 MB | **A** The trained EfficientNetB0 classifier (101 classes). Copy of `model_outputs/food101_nutrition_final.keras`. Loaded with `compile=False` at runtime |
| `classifier/class_names.json` | 1.5 KB | **A** The 101 Food-101 class labels (copy of the one in `model_outputs/`) |
| `food_detector/food_detector.pt` | 6.6 MB | **A** Ultralytics YOLO food detector (113-class, Nutrition5k-style, **not** COCO). Used only to find boxes; never weight. Copy of root `yolov8n.pt` |

## `model_outputs/` (5 files, ~136 MB) - original training outputs

| File | Size | Role |
|---|---|---|
| `notebook82076030c6 (1).ipynb` | 952 KB | **A/D** The **real training notebook** (48 cells) that produced the metrics. Ran on Kaggle, 2× Tesla T4, Python 3.13.15, TF 2.20.0 |
| `food101_nutrition_final.keras` | 52.5 MB | **A** Final trained model (== `models/classifier/classifier.keras`). The model whose metrics were reported |
| `best_food101_finetuned.keras` | 52.5 MB | **A** Best-validation-epoch checkpoint of stage 2 (same weights family as above, separate serialization, different bytes) |
| `best_food101_stage1.keras` | 29.9 MB | **A** Stage-1 (head-only) best checkpoint - keep for reference, not used at runtime |
| `class_names.json` | 1.5 KB | **A** 101 class labels (copy in `models/classifier/`) |

## `notebooks/` (1 file)

| File | Size | Role |
|---|---|---|
| `kaggle_weight_macros.ipynb` | 6.3 KB | **D** Thin Kaggle orchestrator: adds `src/` to the path, loads config, resolves inputs, optionally trains, analyzes a sample image, and launches the Gradio app. Importable anywhere |

## `scripts/` (5 scripts + 6 compiled caches)

| File | Size | Role |
|---|---|---|
| `inspect_inputs.py` | 1.2 KB | **R** Print environment report + what datasets/models were discovered (`--config`). First command to run after cloning |
| `run_analysis.py` | 2.5 KB | **R** Analyze one food image. Flags: `--measured`, `--top-k`, `--model-dir`, `--regressor-dir`, `--config`, `--json` |
| `run_multi_analysis.py` | 4.2 KB | **R** Analyze a multi-dish photo. Flags: `--measured`, `--dish-grams`, `--top-k`, `--max-dishes`, `--crop-padding`, `--model-dir`, `--detector-weights`, `--nutrition-csv`, `--config`, `--json` |
| `smoke_test.py` | 3.9 KB | **R** Synthetic end-to-end check: tiny train → save → reload → analyze. No external data needed |
| `train_classifier.py` | 2.5 KB | **R** Train the classifier on discovered images. Flags: `--config`, `--full` (full run vs quick) |
| `__pycache__/` (6 .pyc) | G | Compiled bytecode |

## `src/foodmacros/` (31 .py files + 31 caches, ~0.4 MB)

### Package root & config

| File | Role |
|---|---|
| `__init__.py` | Public exports + `__version__ = "0.1.0"` |
| `config.py` | `AppConfig` dataclass tree; strict JSON loader (rejects unknown keys); `PROJECT_ROOT` |
| `env_checks.py` | Environment report (Python/TF/GPU/packages), mixed-precision setup, `default_working_dir()` |

### `data/`

| File | Role |
|---|---|
| `__init__.py` | Exports |
| `discovery.py` | Scans `/kaggle/input` and local `data/` for images, nutrition CSV, Nutrition5k markers, `.pt` weights; returns `ResolvedInputs` with `missing` guidance |
| `splits.py` | Canonical class ordering, per-class train/val/test splits, overlap assertions, saves `class_names.json` |
| `preprocessing.py` | Robust image loading/decoding + `tf.data` builders |

### `classification/`

| File | Role |
|---|---|
| `__init__.py` | Exports |
| `model.py` | EfficientNetB0 builder (+ optional keras-hub ViT fusion, off by default), label-smoothing loss, fine-tuning helpers |
| `train.py` | Two-stage training (frozen head → partial unfreeze), portable artifact saving (`.keras`, class names, mapping, config) |
| `evaluate.py` | Top-1/top-5 accuracy, classification report, confusion matrix |
| `predict.py` | Robust single-image prediction (`compile=False` reload) |

### `nutrition/`

| File | Role |
|---|---|
| `__init__.py` | Exports |
| `csv_schema.py` | Column validation/coercion + invalid-row accounting |
| `lookup.py` | `NutritionTable.per_100g(label, strategy)` with documented multi-row strategies (`nearest_weight` default, `per_gram_mean`, `interpolate`) and display-name aliases |
| `portion.py` | Scale per-100 g values to a chosen gram amount, with rounding |

### `weight/`

| File | Role |
|---|---|
| `__init__.py` | Exports |
| `estimator.py` | Trust order: measured > estimated > experimental > unavailable, with a reliable-range check |
| `regression.py` | Nutrition5k metadata parsing + mass/macro regressor (needs the ~23 GB mirror to train) |
| `depth_model.py` | Experimental depth-estimation model (disabled by default) |
| `calibration.py` | Experimental depth calibration helpers |
| `volume.py` | Experimental volume-from-depth helpers |
| `density.py` | Food density access (reads `food_densities.json`) |

### `detection/`

| File | Role |
|---|---|
| `__init__.py` | Exports |
| `yolo_detector.py` | Guarded Ultralytics wrapper: checks package present, verifies class names; failures become a clear `load_error`. Returns boxes only |
| `crop.py` | Loads image at native resolution and crops detected boxes (optional padding) |
| `visualize.py` | Draws numbered boxes for the UI |

### Orchestration & reporting

| File | Role |
|---|---|
| `pipeline.py` | `FoodAnalysisPipeline`: `from_discovery`/`from_artifacts` (defaults to bundled `models/classifier` + `models/food_detector`), `analyze()` runs stages 1-8, returns `AnalysisResult` |
| `multi_pipeline.py` | `MultiDishPipeline`: detect → crop → classify each → nutrition per dish; `apply_weights(result, per_dish_grams, include)` recomputes totals without re-detecting |
| `reporting.py` | `format_analysis()` (text), `analysis_to_dict()` (JSON), shared `LIMITATIONS` |
| `multi_reporting.py` | `format_multi_analysis()`, `multi_analysis_to_dict()` |

## `tests/` (21 test files + `conftest.py` + 22 caches, ~0.46 MB)

`conftest.py` adds `src/` to `sys.path` (so `pytest` works with no setup) and
provides fixtures: `image_factory`, `synthetic_dataset`, `synthetic_nutrition_csv`.

| File | Tests | Covers |
|---|---|---|
| `test_config.py` | 7 | Strict JSON loading, typo rejection, defaults |
| `test_discovery.py` | 9 | Finding images/CSV/Nutrition5k/`.pt`; missing-input messages |
| `test_splits.py` | 8 | Split correctness, overlap safety, class-name persistence |
| `test_preprocessing.py` | 8 | Image decode + `tf.data` builders |
| `test_classification_model.py` | 6 | Model build, ViT fusion opt-out |
| `test_classification_train.py` | 1 | Training happy path + artifact writing |
| `test_classification_evaluate.py` | 5 | Top-1/top-5, report helpers |
| `test_classification_predict.py` | 6 | Single-image prediction robustness |
| `test_nutrition.py` | 20 | Lookup strategies, aliases, missing labels |
| `test_portion.py` | 6 | Scaling/rounding |
| `test_weight_estimator.py` | 12 | Trust ordering + reliable range |
| `test_weight_regression.py` | 8 | Nutrition5k parsing/regressor |
| `test_weight_geometry.py` | 13 | Geometry helpers |
| `test_weight_depth_model.py` | 3 | Depth-model guards |
| `test_detection.py` | 9 | YOLO wrapper guards (package missing, class-name verify) |
| `test_detection_crop.py` | 9 | Full-res cropping + padding |
| `test_pipeline.py` | 14 | End-to-end analyze; default artifact resolution; honest fallbacks |
| `test_multi_pipeline.py` | 12 | Multi-dish analyze; `apply_weights` |
| `test_app.py` | 8 | Gradio helpers, missing-gradio handling, multi-compute totals |
| `test_scripts.py` | 5 | CLI argument handling (missing image, `--json`) |
| `test_env_checks.py` | 4 | Environment report / mixed-precision helpers |

**Total: 173 test functions** across the 21 files above (171 pass, 2 skip).

## `.pytest_cache/` (5 files, generated) - G

`CACHEDIR.TAG`, `README.md`, `.gitignore`, `v/cache/lastfailed`,
`v/cache/nodeids` — pytest's cache of last-run test IDs. Regenerated by pytest.

---

## New files added by this documentation task (8)

| File | Effect |
|---|---|
| `backups/README.md` | Copy of the original README before rewrite |
| `README.md` | Rewritten (replaces the original) |
| `docs/PROJECT_GUIDE.md` | New guide |
| `docs/FILE_BY_FILE_REFERENCE.md` | This file |
| `docs/MODELS_DATASETS_AND_METRICS.md` | New guide |
| `docs/NOTEBOOK_GUIDE.md` | New guide |
| `docs/RUN_AND_TROUBLESHOOT.md` | New guide |
| `docs/CLEANUP_AUDIT.md` | New guide |
| `docs/FRIEND_HANDOVER.md` | New guide |

No existing notebook, dataset, or trained-artifact file was modified.