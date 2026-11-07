# Food Image-Based Weight and Macronutrient Estimation

A final-year project: given a photo of food, predict what dish it is, optionally
estimate the portion weight, and report the macronutrients (calories, protein,
carbs, fat, and more) for that portion.

It handles a **single dish** and a **multi-dish plate** (several foods in one
photo, detected and analyzed separately).

| Interface | What it does |
|---|---|
| `app/gradio_app.py` | A dashboard UI with **Single Dish**, **Multi-Dish**, and **About** navigation |
| `scripts/run_analysis.py` | Command-line analysis of one food image |
| `scripts/run_multi_analysis.py` | Command-line analysis of a photo with several dishes |
| `src/foodmacros` | Reusable Python package used by all of the above |

## Guiding principle: honesty over guessing

The system **never fabricates** a weight, a macro value, a confidence, or an
accuracy. When a source is missing it says so explicitly (`unavailable` + a
reason) and degrades cleanly.

| Capability | Requires | If missing |
|---|---|---|
| Dish classification (top-k) | a trained `classifier.keras` | says `unavailable` with a reason |
| Food detection (bounding boxes) | `ultralytics` + the bundled food-model weights | reports exactly what is missing |
| Portion weight | a user measurement or a validated Nutrition5k regressor | reports `unavailable`; **never invents grams** |
| Macronutrients | `data/nutrition.csv` (Food-101 nutrition data) | reports `unavailable`; a *reference* quantity is clearly labelled |

Weight is reported as exactly one of:

- `measured` — user-supplied grams (the only value treated as ground truth);
- `estimated` — Nutrition5k regression, only accepted inside a reliable range;
- `experimental` — depth × density, disabled by default;
- `unavailable` — nothing reliable; **no grams are shown**.

A fixed serving (default 100 g) is only a *nutrition reference*, never the
predicted weight. A plate total in the multi-dish UI is shown only when every
included dish has a usable weight **and** nutrition; otherwise the total is
`unavailable` with a reason. See `docs/RUN_AND_TROUBLESHOOT.md`.

## Quick start

```bash
# 1. Install core dependencies (Python, NumPy, pandas, sklearn, TensorFlow, Pillow)
pip install -r requirements.txt

# 2. Optional extras for testing / UI / detection
pip install -r requirements-dev.txt
pip install gradio ultralytics

# 3. Sanity-check that the whole project runs on synthetic data
python scripts/smoke_test.py

# 4. Analyze an image (single dish)
python scripts/run_analysis.py path/to/meal.jpg --measured 250

# 5. Analyze a multi-dish photo (uses the bundled detector + classifier)
python scripts/run_multi_analysis.py path/to/plate.jpg --dish-grams "200,100,50"

# 6. Launch the web app
python app/gradio_app.py            # add --share for a public temporary link
```

> The classifier recognizes the **101 Food-101 dishes**. It reaches **74.05%
> top-1 accuracy on a held-out test set**, as measured by the training notebook
> — so a wrong prediction is expected for roughly 1 in 4 single-dish photos.
> Confidence is shown, but confidence is *not* a guarantee of correctness. See
> `docs/MODELS_DATASETS_AND_METRICS.md`.

## Project map

```
Final_Year_Project/
  configs/                 JSON settings (default.json, multi_dish.json, food_densities.json)
  data/nutrition.csv       nutrition table (101 Food-101 classes, 5 weight rows each)
  models/                  runtime artifacts (classifier, food detector, mass regressor)
  model_outputs/           original training outputs + the big training notebook
  notebooks/               lightweight Kaggle orchestrator
  src/foodmacros/          the reusable package (classification, detection, nutrition, weight, pipeline)
  app/gradio_app.py        the web UI
  scripts/                 command-line tools
  tests/                   pytest suite (194 test functions; 192 pass, 2 skipped)
  docs/                    user guides (see below)
```

## Documentation

The `docs/` folder is written for a beginner who is new to the project:

| File | Purpose |
|---|---|
| `docs/PROJECT_GUIDE.md` | Big picture: how the project works, module by module |
| `docs/FILE_BY_FILE_REFERENCE.md` | Every file in the repository, one line each |
| `docs/MODELS_DATASETS_AND_METRICS.md` | Datasets, model architecture, and the **real** accuracy numbers |
| `docs/NOTEBOOK_GUIDE.md` | The three notebooks and what each cell does |
| `docs/RUN_AND_TROUBLESHOOT.md` | Exact commands to install, run, and debug |
| `docs/CLEANUP_AUDIT.md` | What could be cleaned up (a proposal — nothing is deleted without approval) |
| `docs/FRIEND_HANDOVER.md` | A short, submission-ready description of the project |

## Testing

```bash
python -m pytest tests -q
```

The suite has **194 test functions** across 26 test files and currently reports
**192 passed, 2 skipped** (`tests/conftest.py` puts `src/` on the path
automatically, so no extra setup is needed).

## What is verified locally vs. what is not

Verified in this development environment:

- The full pytest suite passes (192 passed, 2 skipped).
- `scripts/smoke_test.py` trains a tiny classifier, saves it, reloads it through
  the pipeline, and analyzes an image end to end.
- The bundled food detector loads and detects real food; the classifier assigns
  Food-101 labels with real nutrition lookup for all 101 classes.
- The bundled mass regressor loads and returns `estimated` weights for
  single-dish and per-dish multi-dish analysis; a measured weight overrides it.
- The Gradio app launches and serves at `http://127.0.0.1:7860`.
- Single- and multi-dish CLI analysis run on real Food-101 images and composite
  plates, including the honest `unavailable` fallbacks.

Not verified here (needs a dataset/GPU this machine does not have):

- Re-training on the full Food-101 dataset and re-measuring accuracy locally.
- Re-training the Nutrition5k mass regressor (the ~23 GB dataset mirror is not
  attached); the trained artifacts are bundled and used at inference.
- Depth-based weight estimation (disabled by default, experimental).
- A public Gradio share link or Hugging Face deployment.

More detail is in `docs/RUN_AND_TROUBLESHOOT.md` and
`docs/PROJECT_GUIDE.md`.

## Weight estimation (RGB-only, Nutrition5k)

This project supports **RGB-only food weight estimation** using the Nutrition5k dataset.

### How it works
```
Nutrition5k RGB image (rgb.png)
          +
Nutrition5k measured total_mass (from dish_metadata_cafe1.csv / dish_metadata_cafe2.csv)
          ↓
   dataset construction (matched by dish_id)
          ↓
    train/val/test split by dish_id (no leakage)
          ↓
     MobileNetV2 regression (ImageNet pretrained)
          ↓
     estimated mass in grams
```

- **Labels come from Nutrition5k**: `total_mass` in grams from metadata files, joined by `dish_id` to `imagery/realsense_overhead/<dish_id>/rgb.png`. Only `rgb.png` is used (no depth/MiDaS).
- **RGB-only baseline**: This is the first experiment (`RGB → mass_g`). Depth/MiDaS can be added later for comparison.
- **Dish-level splitting**: Splits are done by unique `dish_id` to prevent leakage between train/validation/test sets.
- **Normalization**: Targets are normalized using training split statistics only; predictions are denormalized back to grams.

### Bundled model and measured accuracy
The trained RGB regressor ships in `models/weight/` (tracked with Git LFS):
`mass_regressor.keras`, `normalization.json`, `config.json` and
`test_metrics.json`. On the held-out Nutrition5k test split (n=489) it scores:

| Metric | Value |
|---|---|
| MAE | 71.87 g |
| RMSE | 103.59 g |
| R² | 0.777 |
| MAPE | 80.29% |
| Mean-baseline MAE | 174.90 g |

This is a useful estimate, **not a measurement**: MAPE is high because small
dishes dominate the relative error. Predictions are only accepted inside the
configured reliable range (5–2000 g).

### Download Nutrition5k
Download from the [Nutrition5k GCS bucket](https://console.cloud.google.com/storage/browser/nutrition5k_dataset/nutrition5k_dataset). Recommended CLI:
```bash
gsutil -m cp -r gs://nutrition5k_dataset/nutrition5k_dataset/ ./nutrition5k/
# or
gcloud storage cp -r gs://nutrition5k_dataset/nutrition5k_dataset/ ./nutrition5k/
```

Expected layout:
```
nutrition5k/
├── imagery/realsense_overhead/dish_*/rgb.png
└── metadata/dish_metadata_cafe1.csv, dish_metadata_cafe2.csv
```

Point to it via `--nutrition5k /path/to/nutrition5k` or set `data.nutrition5k_root` in config.

### Train
```bash
python scripts/train_weight.py --nutrition5k /path/to/nutrition5k --output models/weight
```

This will:
1. Build the dataset CSV from metadata + rgb.png (reporting counts of metadata dishes, images found, valid pairs, skipped)
2. Create dish-level train/validation/test splits (70/15/15) in `artifacts/weight_dataset/`
3. Train MobileNetV2 mass regressor (grams only), save best model
4. Evaluate on test set (MAE/RMSE/R²/MAPE in grams), compare to mean-baseline, generate plots in `artifacts/weight_evaluation/`
5. Save `models/weight/mass_regressor.keras`, `normalization.json`, `config.json`, `test_metrics.json`

### Predict on a new image
```bash
python scripts/predict_weight.py --image /path/to/food.jpg --model models/weight
```
Output: `Estimated food weight: XXX.X g`. No Nutrition5k metadata or depth required at inference.

### Integration
The bundled artifacts in `models/weight` are picked up automatically when
`weight.enable_regression` is true (the default), so neither the CLI nor the
Gradio app needs an extra flag. Both pipelines use it:

- **Single Dish**: one prediction for the whole image.
- **Multi-Dish**: one prediction per detected dish crop; the plate total is
  computed from those estimates, and a per-dish measured weight overrides only
  that dish.

You can also point at another directory explicitly with
`FoodAnalysisPipeline.from_artifacts(..., mass_regressor_dir=...)`,
`MultiDishPipeline.from_defaults(..., mass_regressor_dir=...)`, or
`python app/gradio_app.py --regressor-dir <dir>`. The predicted grams are passed
to the nutrition calculator for scaling; a user-supplied measured weight always
takes precedence.

## A note on the training notebook

The real training run lives at
`model_outputs/notebook82076030c6 (1).ipynb` (ran on Kaggle with 2× Tesla T4
GPUs, TensorFlow 2.20). `Food_101_Nutrition_Fixed.ipynb` at the project root is
an **unexecuted** editor copy of
`backups/Food_101_Nutrition_Fixed.backup.ipynb` - do not open it expecting
results. Read `docs/NOTEBOOK_GUIDE.md` before opening any notebook.