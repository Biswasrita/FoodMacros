# Architecture

The project is a set of independent, individually-testable modules orchestrated
by `pipeline.py`. Each module degrades cleanly when its inputs are missing.

## Data flow

```
image ──▶ discovery ──▶ pipeline.analyze ──▶ reporting
                          │
                          ├─ classification  (classifier.keras) → top-k labels
                          ├─ detection       (optional YOLO)      → boxes (never weight)
                          ├─ weight          (measured | regression | depth | unavailable)
                          └─ nutrition       (nutrition.csv)      → macros for the weight
```

Multi-dish flow (`multi_pipeline.py`, for a plate with several foods):

```
image ──▶ detect boxes ──▶ crop each (full-res) ──▶ classify each ──▶ nutrition each
                                                          └─▶ per-dish report + honest plate totals
                                                                  ▲
                          per-dish weights ──▶ apply_weights() ─────┘  (recompute without re-detecting)
```

If detection is unavailable the whole image is treated as a single dish (with a
note). A single measured weight is only applied when exactly one dish is present,
because one number cannot be split between dishes; for several dishes, weights
are supplied per dish. A plate total is shown only when every included dish has a
usable weight and nutrition.

## Modules

- **config** (`config.py`) — one `AppConfig` dataclass tree; strict JSON loader
  that rejects unknown keys so typos fail loudly. Input paths default to `None`
  and are discovered at runtime.
- **env_checks** (`env_checks.py`) — environment report (Python/TF/GPU/packages),
  mixed-precision setup, and `default_working_dir()` for outputs.
- **data**
  - `discovery.py` — scans `/kaggle/input` (and local `data/`) for images,
    the nutrition CSV (with header validation), Nutrition5k markers, and `.pt`
    weights; returns `ResolvedInputs` with actionable `missing` messages.
  - `splits.py` — canonical class ordering, per-class train/val/test splits,
    overlap assertions, class-name persistence.
  - `preprocessing.py` — robust image loading/decoding + `tf.data` builders.
- **nutrition**
  - `csv_schema.py` — column validation/coercion and invalid-row accounting.
  - `lookup.py` — `NutritionTable.per_100g(label, strategy)` with documented
    multi-row strategies (`nearest_weight` default, `per_gram_mean`, `interpolate`).
  - `portion.py` — scaling per-100 g to a chosen gram amount, rounding.
- **classification**
  - `model.py` — EfficientNetB0 builder (+ optional keras-hub ViT fusion),
    preprocessing, a label-smoothing loss (Keras 3 has none for sparse CE),
    fine-tuning helpers.
  - `train.py` — two-stage training (frozen head → partial unfreeze), portable
    artifact saving (`.keras`, class names, label mapping, config).
  - `evaluate.py` — top-1/top-5, classification report, confusion matrix.
  - `predict.py` — robust single-image prediction (`compile=False` reload).
- **weight**
  - `estimator.py` — trust ordering: measured > estimated > experimental >
    unavailable, with a configurable reliable range.
  - `regression.py` — Nutrition5k metadata parsing + mass/macro regressor.
  - `depth_model.py`, `calibration.py`, `volume.py`, `density.py` — the
    experimental depth path (disabled by default).
- **detection** (`yolo_detector.py`, `crop.py`, `visualize.py`) — guarded
  Ultralytics wrapper that checks for the package and verifies class names;
  failures become `load_error`, and boxes are never treated as weight. `crop.py`
  loads images at native resolution and crops detected boxes (with optional
  padding); `visualize.py` draws numbered boxes for the UI.
- **pipeline** (`pipeline.py`) — `FoodAnalysisPipeline.analyze()` runs stages
  1–8 and returns an `AnalysisResult`.
- **multi_pipeline** (`multi_pipeline.py`) — `MultiDishPipeline.analyze()`
  detects dishes, crops each from the full-resolution image, classifies every
  crop, and looks up nutrition per dish. `apply_weights(result, per_dish_grams,
  include)` recomputes macros and the plate total from user-supplied per-dish
  weights without re-running detection. A plate total is only produced when every
  included dish has a usable weight and nutrition.
- **reporting** (`reporting.py`) — `format_analysis()` (text) and
  `analysis_to_dict()` (JSON), sharing one `LIMITATIONS` string.
- **multi_reporting** (`multi_reporting.py`) — `format_multi_analysis()` and
  `multi_analysis_to_dict()` for the multi-dish result.

## Design rules

1. No hard-coded machine paths; discover at runtime.
2. No fabricated values — missing → explicit `unavailable`/reason.
3. Fixed serving is a labelled *reference*, never a predicted weight.
4. Multi-row nutrition is resolved by a documented rule, never "first row wins".
5. Broad `except` is avoided; failures carry a clear message.
