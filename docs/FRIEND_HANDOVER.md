# Friend Handover: a one-page description of this project

This page is written so that anyone - a classmate, a supervisor, an evaluator -
can understand the project quickly and re-run it without digging through code.

---

## What this is

A **food-photo analyzer**. You give it a picture of food and it tells you (1)
what dish it is, (2) how much it weighs (if you provide the number or a
regressor is available), and (3) the calories / protein / carbs / fat for that
portion. It understands **101 common food classes** (Food-101) and can also
handle a **single photo with several different dishes** on one plate, by
detecting each dish, cropping it, and analyzing it separately.

The core idea that makes it defensible: **honesty over guessing.** The system
never makes up a weight, a macro, or an accuracy. When something is missing it
prints `unavailable` with a reason instead of an invented number.

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
- **Weight:** taken from the user if provided (`measured`), otherwise reported
  as `unavailable`. Regression and experimental depth paths exist but are off
  unless their datasets are attached.

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

Multi-dish photo? `python scripts/run_multi_analysis.py plate.jpg --dish-grams "200,100,50"`.

Tests: `python -m pytest tests -q` → **171 passed, 2 skipped**.

## How the code is organized

- `src/foodmacros/` - the reusable package. Small, independent modules:
  `classification/`, `detection/`, `nutrition/`, `weight/`, `data/`, plus
  `config.py`, `pipeline.py`, `multi_pipeline.py`, `reporting.py`.
- `app/gradio_app.py` - the web UI (two tabs: Single dish / Multiple dishes).
- `scripts/` - CLI tools: `inspect_inputs`, `smoke_test`, `run_analysis`,
  `run_multi_analysis`, `train_classifier`.
- `configs/` - JSON settings. Everything starts from `default.json`.
- `models/` - the runtime artifacts (classifier + detector + class names).
- `data/nutrition.csv` - the nutrition table.
- `model_outputs/` - the original training outputs and **the** training
  notebook.
- `tests/` - 173 pytest functions.

## What is verified and what is not

- Verified here: the whole pipeline runs (tests + smoke test), the detector
  finds real food, classification + nutrition work end to end, and the app
  launches locally.
- Not verified here: full re-training on Food-101 (needs the dataset + GPU),
  Nutrition5k weight regression (needs a ~23 GB mirror), depth-based weight
  (needs RGB-D data; off by default), and public hosting of the app.

## A good 10-minute demo for an evaluator

1. `python scripts/smoke_test.py` - shows the whole system working on synthetic
   data ("Smoke test PASSED").
2. `python scripts/run_multi_analysis.py some_plate.jpg --dish-grams "200,100"` -
   shows per-dish classification + a plate total.
3. `python app/gradio_app.py` - the two-tab UI.
4. `python -m pytest tests -q` - the test suite.

## Where everything else lives

Full details: `docs/PROJECT_GUIDE.md`, `docs/FILE_BY_FILE_REFERENCE.md`,
`docs/MODELS_DATASETS_AND_METRICS.md`, `docs/NOTEBOOK_GUIDE.md`,
`docs/RUN_AND_TROUBLESHOOT.md`, `docs/CLEANUP_AUDIT.md`.