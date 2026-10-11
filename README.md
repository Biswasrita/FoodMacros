# Food Image-Based Weight and Macronutrient Estimation

A final-year project: given a photo of food, predict what dish it is, optionally
estimate the portion weight, and report the macronutrients (calories, protein,
carbs, fat, and more) for that portion.

It handles a **single dish** and a **multi-dish plate** (several foods in one
photo, detected and analyzed separately).

| Interface | What it does |
|---|---|
| `app/gradio_app.py` | A friendly web UI with a **Single dish** tab and a **Multiple dishes** tab |
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
  models/                  runtime artifacts (classifier.keras, class_names.json, food_detector.pt)
  model_outputs/           original training outputs + the big training notebook
  notebooks/               lightweight Kaggle orchestrator
  src/foodmacros/          the reusable package (classification, detection, nutrition, weight, pipeline)
  app/gradio_app.py        the web UI
  scripts/                 command-line tools
  tests/                   pytest suite (173 test functions; 171 pass, 2 skipped)
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

The suite has **173 test functions** across 21 test files and currently reports
**171 passed, 2 skipped** (`tests/conftest.py` puts `src/` on the path
automatically, so no extra setup is needed).

## What is verified locally vs. what is not

Verified in this development environment:

- The full pytest suite passes (171 passed, 2 skipped).
- `scripts/smoke_test.py` trains a tiny classifier, saves it, reloads it through
  the pipeline, and analyzes an image end to end.
- The bundled food detector loads and detects real food; the classifier assigns
  Food-101 labels with real nutrition lookup for all 101 classes.
- The Gradio app launches and serves at `http://127.0.0.1:7860`.
- Single- and multi-dish CLI analysis run on real Food-101 images and composite
  plates, including the honest `unavailable` fallbacks.

Not verified here (needs a dataset/GPU this machine does not have):

- Re-training on the full Food-101 dataset and re-measuring accuracy locally.
- Nutrition5k mass regression (the ~23 GB mirror is not attached).
- Depth-based weight estimation (disabled by default, experimental).
- A public Gradio share link or Hugging Face deployment.

More detail is in `docs/RUN_AND_TROUBLESHOOT.md` and
`docs/PROJECT_GUIDE.md`.

## A note on the training notebook

The real training run lives at
`model_outputs/notebook82076030c6 (1).ipynb` (ran on Kaggle with 2× Tesla T4
GPUs, TensorFlow 2.20). `Food_101_Nutrition_Fixed.ipynb` at the project root is
an **unexecuted** editor copy of
`backups/Food_101_Nutrition_Fixed.backup.ipynb` - do not open it expecting
results. Read `docs/NOTEBOOK_GUIDE.md` before opening any notebook.