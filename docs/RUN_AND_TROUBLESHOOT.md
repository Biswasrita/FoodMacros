# Run, Install, and Troubleshoot

This is the practical guide: exact commands, expected output, and fixes for the
common problems. It assumes you are in the project root folder
(`Final_Year_Project/`).

---

## 1. Install

Requirements: **Python 3.11+** (this project was verified on Python 3.11.9).
A GPU is **not** needed to run analysis - only recommended for re-training.

```bash
# Core packages (classification + nutrition): numpy, pandas, scikit-learn,
# tensorflow, Pillow - minimum versions only.
pip install -r requirements.txt

# Optional but recommended:
pip install -r requirements-dev.txt   # pytest + nbformat (for tests)
pip install gradio                     # the web UI
pip install ultralytics                # YOLO food detection (multi-dish mode)

# Only for optional features you probably do not need:
pip install keras-hub                  # optional ViT fusion backbone (off by default)
pip install torch                      # used by some optional weight/depth helpers
```

After installing, verify the environment and that the bundled artifacts are
found:

```bash
python scripts/inspect_inputs.py
```

You should see lines like:

```
Python            : 3.11.9
TensorFlow        : 2.20.0
...
data/nutrition.csv    : found
models/classifier     : found
models/food_detector  : found
```

If something is reported missing, the summary will say exactly what and why -
the project never guesses.

---

## 2. The commands

### Sanity check (no external data needed)

```bash
python scripts/smoke_test.py
```

This trains a *tiny* classifier on synthetic images, saves it, reloads it
through the real pipeline, and analyzes an image. It prints a report and ends
with:

```
Smoke test PASSED
```

If this works, the project is installed correctly. (Its numbers such as
"bibimbap 33.4%" are toy plumbing - not real accuracy.)

### Analyze a single dish

```bash
python scripts/run_analysis.py path/to/meal.jpg
python scripts/run_analysis.py path/to/meal.jpg --measured 250
python scripts/run_analysis.py path/to/meal.jpg --json
python scripts/run_analysis.py path/to/meal.jpg --top-k 5
```

Notes:

- `--measured G` adds the portion weight you weighed; macros are then scaled to
  it. Without it, macros use the 100 g **reference** quantity (clearly labelled).
- `--json` prints the structured result (useful for scripting).
- No machine-specific paths are needed - the bundled `models/` and
  `data/nutrition.csv` are found automatically.

### Analyze a multi-dish photo

```bash
python scripts/run_multi_analysis.py path/to/plate.jpg
python scripts/run_multi_analysis.py path/to/plate.jpg --dish-grams "200,100,50"
python scripts/run_multi_analysis.py plate.jpg --max-dishes 3 --crop-padding 0.1
python scripts/run_multi_analysis.py path/to/plate.jpg --json
```

Notes:

- `--dish-grams "200,100,50"` provides one weight per detected dish **in table
  order**. Leave a slot blank to skip that dish, e.g. `"200,,50"`.
- One `--measured` weight for the whole plate is used **only** when exactly one
  dish is detected.
- A **plate total** is printed only when every included dish has both a weight
  and nutrition. Otherwise you will see `unavailable (missing weight for dish 2)`.

Real verified example on a composite plate:

```
total: 960 kcal / 380 g   (from --dish-grams "200,100,50,30")
```

### Launch the web app

```bash
python app/gradio_app.py
python app/gradio_app.py --port 7861
python app/gradio_app.py --share      # temporary public link
```

- Opens **Single dish** and **Multiple dishes** tabs at `http://127.0.0.1:7860`.
- The multi-dish tab is a two-step flow: **1) Detect & classify**, then fill in
  the **Grams (g)** column and click **2) Compute nutrition & total**.
- `--share` prints a temporary public URL (valid ~72h per Gradio) - useful for
  demos, not for permanent hosting.

### Run the tests

```bash
python -m pytest tests -q
```

Expected result: **171 passed, 2 skipped** (173 test functions). No extra setup
is needed - `tests/conftest.py` puts `src/` on the path for you.

### Re-train the classifier (optional, needs the dataset + GPU)

```bash
python scripts/train_classifier.py            # honors config mode (quick by default)
python scripts/train_classifier.py --full     # full training run
```

- Uses the discovered dataset (`config.data.image_root` or `/kaggle/input`).
- With no dataset it prints guidance: attach `hari31416/food-101` or set
  `config.data.image_root`.
- `--full` sets `runtime.mode = full` (the two-stage schedule from
  `configs/default.json`). Expect many hours on a single GPU.

---

## 3. Expected accuracy on your photos

- Single dish: the classifier's **first** guess is right ~74% of the time
  (74.05%); the correct dish is in the **top 5** ~92% of the time (91.69%).
  So about **1 in 4 photos will have a wrong first guess** - this is the model,
  not a bug. Look at the top-k list.
- Multi-dish: each crop is classified independently, so the same caveat applies
  **per dish**. If the plate has several foods in one photo without clear
  separation, the detector may crop combined regions - check the numbered boxes.
- Confidence is shown but is **not** a guarantee. A dish can be predicted at
  99% and still be wrong.

---

## 4. Troubleshooting

| Symptom | Cause & fix |
|---|---|
| `Image not found: ...` | Wrong path. Use the full file name; check for typos and spaces (quote the path on Windows/Linux shells). |
| `module gradio not installed` | Run `pip install gradio`. The app degrades gracefully and will explain this. |
| `detection unavailable: ultralytics ...` | Run `pip install ultralytics`. Single-dish analysis still works without it. |
| Multi-dish says `total: unavailable (missing weight for dish N)` | The dish at position N has no weight. Pass `--dish-grams "…,WN,…"` or edit the **Grams (g)** column in the UI. This is intended honest behavior. |
| `No image dataset found; attach 'hari31416/food-101'...` (training only) | You need the Food-101 dataset to train. `run_analysis` does **not** need it. |
| `mass regressor ... unavailable` / weight shows `unavailable` | Expected: Nutrition5k (the ~23 GB mirror) is not attached, so weights are only `measured` or `unavailable`. Never a bug. |
| Wrong dish predicted on my photo | Normal for ~26% of first guesses (74% top-1). Check the top-k list, try a more focused crop, or use multi-dish if several foods share the frame. |
| Classifier file missing (`models/classifier/classifier.keras`) | The bundled model was removed or not copied. Restore it from `model_outputs/food101_nutrition_final.keras` (byte-identical). |
| `Port 7860 is already in use` | Run `python app/gradio_app.py --port 7861`. |
| Slow first run / `Downloading ... efficientnetb0` | Keras is fetching pretrained ImageNet weights once; subsequent loads are fast. |
| `python` not recognized | Use your Python executable path (e.g. `py`, `python3`) or activate a virtual environment. |
| `--dish-grams "200,,50"` on some shells | The empty slot is represented by two commas with nothing between them; keep the quotes. |

---

## 5. Performance expectations

- **Analysis (inference)** on a laptop CPU: ~1-3 s per image for
  classification; detection adds ~0.5-1 s. The app loads models once at start.
- **Training** on the full Food-101 dataset wants a GPU. The Kaggle run took
  roughly 1.5 h for Stage 1 (15 epochs) plus up to another ~2 h for Stage 2
  (20 epochs) with early stopping, on 2× T4 GPUs with batch size 16.

---

## 6. What this machine can and cannot verify

Can verify (and has been): install, smoke test, tests, single- and multi-dish
CLI on real images, honest `unavailable` fallbacks, Gradio local launch.

Cannot verify on this machine: full Food-101 re-training and local re-measurement
of accuracy (no dataset/GPU), Nutrition5k mass regression (no ~23 GB mirror),
depth estimation (needs RGB-D camera data), and public hosting of the Gradio
app. Those need the resources Kaggle/providers offer - see `NOTEBOOK_GUIDE.md`.