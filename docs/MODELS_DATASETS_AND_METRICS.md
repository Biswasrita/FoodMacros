# Models, Datasets, and Metrics (the real numbers)

This document explains **what the models are**, **what data they used**, and
**exactly which numbers are trustworthy**. Every number below is traceable to
`model_outputs/notebook82076030c6 (1).ipynb` (the training notebook) or to
files in this repository. Nothing has been guessed.

---

## 1. Datasets

### Food-101 (images) - the classification dataset
- **101 classes** of common food, e.g. `pizza`, `sushi`, `apple_pie`, `bibimbap`.
- Obtained via Kaggle dataset `hari31416/food-101` (the 4.66 GB `food-101.zip`
  is available offline at the machine's Downloads folder and mirrors the Celery
  `food-101` dataset).
- The Kaggle copy the notebook used provides a **train** folder (95,950 images,
  950 per class) and a **validation** folder (5,050 images, 50 per class).
- Held-out **test** set: the notebook deliberately carved the *last 25%* of the
  validation folder (the final 79 of 316 batches = **1,264 images**) and never
  trained on it. This test set is *not* the official Food-101 test split
  (that split does not ship with this Kaggle copy), so treat "74.05%" as: *on
  a held-out slice of the validation folder*.

### Nutrition table (macros)
- Kaggle dataset `sanadalali/food-101-nutritional-information` (CC0-1.0).
- Bundled as `data/nutrition.csv`: **505 rows** = 101 labels × 5 different
  serving weights (80, 100, 120, 150, 200 g per label).
- Columns: `label, weight, calories, protein, carbohydrates, fats, fiber,
  sugars, sodium`. Example (`apple_pie`, 100 g): 300 kcal, 3 g protein, 45 g
  carbs, 12 g fat.
- The CSV covers all 101 Food-101 labels. Seven labels appear with a human
  display name in the CSV (e.g. `Cupcakes`, `Fish and Chips`) and are resolved
  to the Food-101 file names (`cup_cakes`, `fish_and_chips`) through small
  aliases in `src/foodmacros/nutrition/lookup.py`.

### Nutrition5k (weight regression) - NOT attached
- A ~23.5 GB mirror would let the project train the **mass regressor** that
  estimates a portion's weight from its photo. It is not present, so the
  weight path reports `estimated` as unavailable and never invents grams.

### Food detector weights
- The detector `models/food_detector/food_detector.pt` is an **ultralytics
  YOLO** model with **113 nutrition-style classes** (Nutrition5k-style class
  names). It is **not** a COCO model, so its class list differs from Food-101;
  it is loaded with `verify_class_names=False`. It was **not trained in this
  project** - it is pretrained weights used as-is, only for finding boxes.

---

## 2. The classifier architecture

- **Backbone:** EfficientNetB0, ImageNet-pretrained, input **224×224**.
- **Head:** GlobalAveragePooling → BatchNorm → Dropout(0.45) →
  Dense(512, ReLU + L2) → BatchNorm → Dropout → Dense(101, softmax).
- **Data augmentation inside the model:** flip, rotation 0.15, zoom 0.20,
  translation 0.10, contrast 0.20, brightness 0.15.
- **Training was two-stage:**
  1. **Stage 1 - feature extraction:** backbone frozen, train only the head at
     lr 1e-3 (cosine decay), up to 15 epochs.
  2. **Stage 2 - fine-tuning:** unfreeze the *last 30* backbone layers (23
     trainable base layers in total), BatchNorm kept frozen, lr 1e-5, up to 20
     epochs.
- **Regularization:** label smoothing 0.1, weight decay (L2) on the head,
  early stopping, `ReduceLROnPlateau`, best-checkpoint saving.
- **Speed trick:** mixed precision (float16 policy).
- **Compute:** ran on Kaggle with **2× Tesla T4 GPUs**, TensorFlow **2.20.0**,
  batch size **16**, Python 3.13.15.
- **Local config:** `configs/default.json` mirrors these choices with a
  `quick` mode (1 short epoch) so it can be smoke-tested without a GPU.

---

## 3. The measured accuracy (the numbers you may quote)

These are printed by the training notebook's `model.evaluate()` cells. They
were measured on the **final** trained model, and that same model was saved to
`food101_nutrition_final.keras` (copied to `models/classifier/classifier.keras`,
the runtime classifier). So the numbers describe the model that ships in
this repository, but they were **not re-measured on this computer** (no GPU /
dataset here).

| Stage | Test set | Top-1 | Top-5 | Loss |
|---|---|---|---|---|
| Stage 1 (head only) - validation | 3,792 images | 64.05% | 87.43% | 1.5294 |
| Final (after fine-tune) - validation | 3,792 images | 71.63% | 91.42% | 1.2189 |
| **Final - held-out TEST** | **1,264 images** | **74.05%** | **91.69%** | **1.1639** |

Notes:

- **Top-1** = the model's single best guess is the correct dish.
- **Top-5** = the correct dish appears somewhere in the model's top 5 guesses.
- Both are standard classification metrics; report both, never only one.
- For a **single photo**, 74% top-1 means the *first guess* is wrong for about
  **1 in 4** photos - that is normal and expected. The correct dish is inside
  the **top-5 about 92%** of the time.
- In **multi-dish** mode every crop is classified independently, so the same
  74%/92% caveat applies **per dish**.

---

## 4. Metrics you should NOT quote

### Precision / recall / F1 per class (and the "1.58%" number)
The notebook contains a per-class `classification_report` cell and a second
cell that prints `Overall accuracy on test set: 1.58%`. **Do not use these.**

They are broken by a *data-ordering bug*, not a model flaw: the test set was
re-shuffled after being defined, so `y_true` and `y_pred` became misaligned
(the printed per-class numbers are effectively random and contradict the 74.05%
from the same notebook). The **trustworthy** number is the `model.evaluate()`
result in the earlier cell - the **74.05% / 91.69%** in the table above.

### mAP (0.775 / "0.802")
There is **no mAP anywhere in this project.** A search of every notebook and
output file found no 0.775 or 0.802. The food detector was neither trained nor
evaluated here, so **no mAP value should be cited** for it. (For background
only: mAP50 is the average-precision at IoU threshold 0.50, mAP50-95 at
0.50-0.95; neither was computed here.)

---

## 5. The food detector and what its confidence means

- File: `models/food_detector/food_detector.pt` (identical to root
  `yolov8n.pt`). Ultralytics YOLO, 113 nutrition-style classes.
- It outputs **boxes** (and a per-box confidence) saying *where* food is.
- Its class names are **not** Food-101, so box labels are *not* used as the
  answer. The pipeline crops each box and lets the **classifier** assign the
  Food-101 label.
- Detection config: confidence threshold 0.25, IoU 0.45 (`configs/default.json`
  and `configs/multi_dish.json`).
- **A high detection box confidence does not mean the dish was classified
  correctly** - classification confidence is a separate number and, like any
  softmax confidence, is *not* a guarantee of correctness.

---

## 6. Runtime artifacts and what each is used for

| Path | Model | Used for |
|---|---|---|
| `models/classifier/classifier.keras` | EfficientNetB0, 101 classes | Predicting the dish |
| `models/classifier/class_names.json` | 101 label names | Turning index → dish name |
| `models/food_detector/food_detector.pt` | YOLO, 113 classes | Finding food boxes (multi-dish + UI) |
| `data/nutrition.csv` | 505 nutrition rows | Macros per dish × weight |

All four are **bundled defaults** - the pipeline loads them automatically, so
`run_analysis.py`, `run_multi_analysis.py`, and the Gradio app work right after
`pip install -r requirements.txt`.

---

## 7. Weight: measured > estimated > experimental > unavailable

| Status | Meaning | When it happens |
|---|---|---|
| `measured` | User-supplied grams (trusted) | `--measured`, `--dish-grams`, or the UI Grams column |
| `estimated` | Nutrition5k regression in a reliable range | Only if a trained regressor + data exist (not here) |
| `experimental` | Depth volume × density | **Off by default**; must be enabled deliberately |
| `unavailable` | No trustworthy number | Everything else; **no grams invented** |

Without a weight, macros are reported for the reference quantity (100 g) and
that is clearly labelled as a *reference*, never as the real portion.

---

## 8. One-paragraph summary you can reuse

> The project classifies food photos into one of 101 Food-101 dishes with an
> EfficientNetB0 trained in two stages (frozen-head extraction then fine-tuning
> of the last 30 layers) on Kaggle with 2× T4 GPUs. On a held-out 1,264-image
> test set it reaches 74.05% top-1 accuracy and 91.69% top-5 accuracy. A
> separate 113-class YOLO food detector finds food boxes so several dishes in
> one photo can each be classified and nutrition-looked-up; masks never become
> weight. Macronutrients come from a 101-class nutrition CSV (505 rows, 5
> serving weights per class). Portion weight is never invented - it is reported
> as user-measured, regression-estimated (not active), experimental depth
> (disabled), or unavailable.