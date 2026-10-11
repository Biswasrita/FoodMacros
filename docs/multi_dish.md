# Multi-Dish Analysis

Analyze a single photo that contains **several foods** on one plate: detect each
dish, crop it, classify it, look up its nutrition, and report per-dish values
plus a plate total.

This is additive — the original single-dish pipeline, scripts, and Gradio tab are
unchanged.

## How it works

1. **Detect** food boxes with the bundled YOLO food detector.
2. **Crop** every box from the *full-resolution* image (with a little padding so
   context is kept).
3. **Classify** each crop with the Food-101 EfficientNetB0 classifier.
4. **Look up** macronutrients for each predicted dish.
5. **Report** per-dish values and an honest plate total.

If detection is unavailable, the whole image is treated as a single dish (with a
note). If a crop cannot be classified, that dish is reported as `unavailable`
with a reason. Nothing is fabricated.

## Artifacts

The pipeline uses these bundled files by default (copies; the originals are left
untouched):

| Purpose | Path |
|---|---|
| Classifier | `models/classifier/classifier.keras` |
| Class names | `models/classifier/class_names.json` |
| Food detector | `models/food_detector/food_detector.pt` |
| Nutrition | `data/nutrition.csv` |

The detector is a 113-class food model (Nutrition5k-style), **not** COCO. Its
class names differ from Food-101, so it is loaded with
`verify_class_names=False`: the detector finds food *boxes* and the classifier
assigns the Food-101 label.

## Usage

CLI:

```
python scripts/run_multi_analysis.py path/to/plate.jpg
python scripts/run_multi_analysis.py path/to/plate.jpg --json
python scripts/run_multi_analysis.py plate.jpg --max-dishes 3 --crop-padding 0.1
python scripts/run_multi_analysis.py plate.jpg --dish-grams "200,100,50,30"
```

`--dish-grams` takes one value per detected dish (in table order). Leave an entry
blank to skip that dish's weight, e.g. `--dish-grams "200,,50"`.

Gradio: `python app/gradio_app.py` opens a **Multiple dishes** tab (alongside the
existing **Single dish** tab). It is a two-step workflow:

1. Upload/capture a photo and click **1) Detect & classify**. This shows the photo
   with **numbered boxes** and a table with one row per dish:
   `Dish | Detected | Predicted | Confidence | Include | Grams (g)`.
2. Edit the **Grams (g)** column for the dishes you weighed and/or untick
   **Include** to leave a dish out, then click **2) Compute nutrition & total**.

The dish table is recomputed without re-running detection.

Python:

```python
from foodmacros.config import load_config
from foodmacros.multi_pipeline import MultiDishPipeline

pipeline = MultiDishPipeline.from_defaults(load_config())
result = pipeline.analyze("plate.jpg")
for dish in result.dishes:
    print(dish.summary())

# Two-phase: detect once, then apply user-provided weights.
result = pipeline.analyze("plate.jpg")
result = pipeline.apply_weights(result, per_dish_grams=[200, 100], include=[True, False])
```

## Weight and totals — what is honest

- Weights are never invented. Each dish's weight comes from either a
  **measured** value (user-provided, e.g. `--dish-grams` or the UI table) or is
  **unavailable**. With no weight, a dish's macros are shown for the reference
  quantity (default 100 g).
- A single **measured plate weight** (`--measured-grams`) describes the *whole
  plate* and is only used when exactly one dish is present. For multiple dishes
  it is ignored with a warning, because one number cannot be split between dishes.
- **Plate totals are only shown when every included dish has a usable weight and
  nutrition.** Otherwise the totals are reported as `unavailable` with a reason
  (e.g. *"missing weight for dish 2"*). This avoids pretending the sum of
  reference-quantity macros is the real plate.
- When shown, `totals_basis` is `per_predicted_weight` and `total_grams` is the
  sum of the weighed portions. Unticking **Include** removes a dish from the
  total (and from `total_grams`) without discarding its per-dish row.

## Status (verified locally)

- The bundled detector loads and detects real food; the classifier assigns
  Food-101 labels; nutrition lookup resolves all 101 labels (including the 7
  display-name aliases such as `cup_cakes` ← `Cupcakes`).
- Verified end to end on a real composite plate: detection produced multiple
  boxes, crops were classified, and per-dish values were reported. With
  `--dish-grams "200,100,50,30"` a plate total of `960 kcal / 380 g` was produced;
  with a blank entry (`--dish-grams "200,,50,30"`) the total correctly became
  `unavailable (missing weight for dish 2)`.
- The Gradio app was launched locally (HTTP 200 at `http://127.0.0.1:7860`) and
  the interface built successfully with the installed Gradio.
- Not verified: accuracy of the detector/classifier on arbitrary real plates —
  run it on your own photos.
