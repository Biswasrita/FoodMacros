# Friend Handover: a one-page description of this project

This page is written so that anyone - a classmate, a supervisor, an evaluator -
can understand the project quickly and re-run it without digging through code.

---

## What this is

A **food-photo analyzer**. You give it a picture of food and it tells you (1)
what dish it is, (2) an estimated or measured weight, and (3) the calories /
protein / carbs / fat for that portion. It understands **101 common food
classes** (Food-101) and can also handle a **single photo with several
different dishes** on one plate, by detecting each dish, cropping it, and
analyzing it separately.

The core idea that makes it defensible: **honesty over guessing.** The system
never presents an estimate as a measurement, and never makes up a macro or an
accuracy. A weight is marked `measured`, `estimated` or `unavailable` with a
reason.

## How it works

- **Classifier:** an `EfficientNetB0` (a standard, ImageNet-pretrained CNN)
  fine-tuned in two stages. First the new classification head is trained while
  the backbone stays frozen, then the last 30 backbone layers are unfrozen and
  trained with a very small learning rate.
- **Detector:** a separate YOLO model finds the food *boxes* in a photo. Boxes
  say where the food is, never how much it weighs. In multi-dish mode each box
  is cropped and classified independently.
- **Nutrition:** a CSV table (101 dishes × 5 serving sizes) maps each predicted
  dish to macros, scaled to the requested grams.
- **Weight:** a user measurement always wins (`measured`). Otherwise a bundled
  MobileNetV2 RGB regressor predicts grams from the photo (`estimated`); in
  multi-dish mode it runs once per dish crop. If neither is available the weight
  is `unavailable` - no grams are invented. The experimental depth path remains
  off by default.

## Results

Measured by the training notebook on a held-out 1,264-image test set:

| Metric | Value |
|---|---|
| Top-1 accuracy | **74.05%** |
| Top-5 accuracy | **91.69%** |
| Test loss | 1.1639 |

Top-1 means "the best single guess is correct"; top-5 means "the correct dish
is among the top 5 guesses". Expect the first guess to be wrong on about 1 in 4
photos - that is a real, captured limitation, not a bug.

Weight regression was measured separately on a held-out Nutrition5k split
(n=489): **MAE 71.87 g, RMSE 103.59 g, R² 0.777, MAPE 80.29%** (mean-baseline
MAE 174.90 g). It is a rough estimate, especially for small dishes; use a
measured weight when accuracy matters.

> Note: two cells of the notebook print a per-class report and a "1.58%"
> number; those cells are broken by a data-ordering bug and should be ignored.
> The trustworthy measurement is the `model.evaluate()` result of 74.05%.

## How to run it in under a minute

```bash
pip install -r requirements.txt
python scripts/inspect_inputs.py       # verify environment
python scripts/run_analysis.py my_photo.jpg --measured 250
python app/gradio_app.py               # optional web UI
```

Command-line analysis estimates weight automatically from the bundled model; pass
`--measured 250` when you can weigh the food. Multi-dish photo?
`python scripts/run_multi_analysis.py plate.jpg` estimates each dish; add
`--dish-grams "200,100,50"` to override selected dishes.

Tests: `python -m pytest tests -q` → **192 passed, 2 skipped**.

## How the code is organized

- `src/foodmacros/` - the reusable package. Small, independent modules:
  `classification/`, `detection/`, `nutrition/`, `weight/`, `data/`, plus
  `config.py`, `pipeline.py`, `multi_pipeline.py`, `reporting.py`.
- `app/gradio_app.py` - the dashboard UI (Single Dish / Multi-Dish / About).
- `scripts/` - CLI tools: `inspect_inputs`, `smoke_test`, `run_analysis`,
  `run_multi_analysis`, `train_classifier`, `predict_weight`, `train_weight`.
- `configs/` - JSON settings. Everything starts from `default.json`.
- `models/` - the runtime artifacts (classifier + detector + mass regressor).
- `data/nutrition.csv` - the nutrition table.
- `model_outputs/` - the original training outputs and **the** training
  notebook (local-only, gitignored).
- `tests/` - 194 pytest functions.

## What is verified and what is not

- Verified here: the whole pipeline runs (tests + smoke test), the detector
  finds real food, classification + nutrition work end to end, the bundled
  weight regressor returns single- and per-dish estimates, and the app launches
  locally.
- Not verified here: full re-training on Food-101 (needs the dataset + GPU),
  **re-training** the Nutrition5k regressor (needs a ~23 GB mirror; inference
  artifacts are bundled), depth-based weight (needs RGB-D data; off by default),
  and public hosting of the app.

## A good 10-minute demo for an evaluator

1. `python scripts/smoke_test.py` - shows the whole system working on synthetic
   data ("Smoke test PASSED").
2. `python scripts/run_multi_analysis.py some_plate.jpg --dish-grams "200,100"` -
   shows per-dish classification + a plate total; run it without `--dish-grams`
   to show automatic per-dish weight estimates.
3. `python app/gradio_app.py` - the two-tab UI.
4. `python -m pytest tests -q` - the test suite.

## Where everything else lives

Full details: `docs/PROJECT_GUIDE.md`, `docs/FILE_BY_FILE_REFERENCE.md`,
`docs/MODELS_DATASETS_AND_METRICS.md`, `docs/NOTEBOOK_GUIDE.md`,
`docs/RUN_AND_TROUBLESHOOT.md`, `docs/CLEANUP_AUDIT.md`.