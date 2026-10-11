# Notebook Guide

There are **three** notebooks in this project, and they play very different
roles. Open the wrong one and you may think nothing ever ran. This guide tells
you which is which and what each cell does.

| Notebook | Status | Role |
|---|---|---|
| `model_outputs/notebook82076030c6 (1).ipynb` | **Executed** (has results) | The real training run that produced the accuracy numbers |
| `Food_101_Nutrition_Fixed.ipynb` (root) | **Not executed** (empty outputs) | Editor copy of the backup; read-only reference |
| `backups/Food_101_Nutrition_Fixed.backup.ipynb` | **Not executed** | Byte-identical backup of the above |
| `notebooks/kaggle_weight_macros.ipynb` | **Not executed** | Lightweight "run the whole project" orchestrator |

---

## 1. `model_outputs/notebook82076030c6 (1).ipynb` - the real training notebook

- **Size:** 952 KB. **Cells:** 48 (12 markdown, 36 code).
- **Environment:** Kaggle, Python **3.13.15**, TensorFlow **2.20.0**, 2× Tesla
  T4 GPUs, mixed float16, batch size 16, resolution 224.
- This is the notebook you quote for accuracy: **74.05% top-1 / 91.69% top-5**
  on a held-out test set.

### What the cells do (grouped)

| Cells | Section | What happens |
|---|---|---|
| 0 | Title | Project description: EfficientNetB0, 2-stage transfer learning, outputs |
| 1-2 | Setup | Install/import helpers |
| 3 | Dataset download | Downloads `food-101.zip` (4.66 GB) from Kaggle |
| 5 | Dataset download | Downloads the nutrition CSV info (6.64 KB, 101 classes) |
| 13-14 | Imports & config | Mixed precision (float16), image size 224, "hold out 25% of validation as a true test set". Prints `TensorFlow Version: 2.20.0` |
| 18 | Data loading | `image_dataset_from_directory`: **95,950** train files / **5,050** validation files / 101 classes; splits the last **79** of 316 validation batches into a held-out test set (1,264 images, 79×16); train 5,997 batches, val 237 batches |
| 19-20 | Augmentation + shapes | In-model augmentation (flip, rotation 0.15, zoom 0.20, translation 0.10, contrast 0.20, brightness 0.15); verifies batch shape `(16, 224, 224, 3)` |
| 22-23 | Model architecture | `EfficientNetB0` (ImageNet) + head: GAP → BN → Dropout(0.45) → Dense(512, ReLU, L2) → BN → Dropout → Dense(101) |
| 25-27 | Stage 1 - feature extraction | Backbone frozen; head trained lr 1e-3 (cosine decay), label smoothing 0.1, up to 15 epochs. `ModelCheckpoint` saves `best_food101_stage1.keras` |
| 28 | Stage 1 evaluation | Validation: loss 1.5294, top-1 **64.05%**, top-5 **87.43%** |
| 29-31 | Stage 2 - fine-tuning | Unfreeze last 30 backbone layers (23/238 trainable), BatchNorm kept frozen, lr 1e-5, up to 20 epochs. Saves `best_food101_finetuned.keras` |
| 32 | **Final evaluation** | Validation: loss 1.2189, top-1 71.63%, top-5 91.42%. Held-out test: loss 1.1639, top-1 **74.05%**, top-5 **91.69%** |
| 34 | Plots | Combined train/val accuracy + loss curves with a "Stage 2 starts" divider |
| 35-37 | Per-class evaluation | ⚠️ **Broken cells - ignore output.** See the bug box below |
| 38-40 | Single-image demo | `predict_food_with_nutrition()` - picks a random test image; the printed demo was `nachos` at **91.39%** with the nutrition row |
| 41-42 | Save | `model.save("food101_nutrition_final.keras")` + `class_names.json` (53 MB) |
| 43-44 | Gradio setup | Builds a small embedded single-image Gradio app |
| 45 | Gradio launch | Ran at `http://127.0.0.1:7860` and printed a **temporary** public URL (expired - do not use it) |
| 46 | Package | Zips the 4 artifacts into `model_outputs.zip` (127 MB) |

### ⚠️ The "1.58%" bug (important to know!)

The notebook contains these two claims that **contradict** cell 32:

- cell 36 prints a per-class `classification_report` full of 0.000 values;
- cell 37 prints `Overall accuracy on test set: 1.58%`.

**Both are wrong, and the correct number is cell 32's 74.05%.** The cause is a
data-ordering bug, not the model: the test dataset is created from a
*shuffled* stream (`shuffle=True`), so the pass that collects true labels and
the pass that runs predictions visit images in different orders. The `y_true`
and `y_pred` arrays no longer line up, so the "report" and the "1.58%" are
computed on scrambled pairs. Never quote "1.58%" - it is an artifact. This is
exactly why `model.evaluate()` on the un-smuggled test set (cell 32) is the
trusted measurement.

---

## 2. `Food_101_Nutrition_Fixed.ipynb` (root) and its backup

- **46 cells** (12 markdown, 34 code), 33 KB.
- **Never executed:** every code cell has `execution_count: null` and empty
  outputs. It is an editor-side "fixed" rewrite kept as a source reference.
- `backups/Food_101_Nutrition_Fixed.backup.ipynb` is **byte-identical** to it.
- Practical advice: if you want the **results**, read notebook #1. Open this one
  only to read the version they intended to fix - there is nothing to run here.

---

## 3. `notebooks/kaggle_weight_macros.ipynb` - the orchestrator

- Tiny (6.3 KB, 9 cells), **never executed**, written for Python 3.11.
- It does not re-train; it runs the packaged code:

| Cell | What it does |
|---|---|
| title | Explains it imports the reusable `src/foodmacros` package |
| 1 | Adds `src/` to the path (Kaggle path first, then repo-relative) and prints whether it is on Kaggle |
| 2 | Loads config, prints the environment report, sets mixed precision, computes the run dir |
| 3 | `resolve_inputs(config)` - shows which datasets/models were found |
| 5 | Optional training block: builds class names, splits, and trains only if a dataset is discovered (`mode=full` recommended) |
| 7 | Builds the single-dish pipeline and analyzes the first discovered image, printing the report |
| 9 | Launches the Gradio app if `gradio` is installed (set `config.ui.share = True` for a public link) |

- To use on Kaggle: put the project at `/kaggle/working/Final_Year_Project`,
  attach `hari31416/food-101` and
  `sanadalali/food-101-nutritional-information`, then Run All with a GPU
  accelerator.

---

## Cheat sheet: "which notebook do I open?"

| Goal | Open |
|---|---|
| See the accuracy numbers / how training worked | `model_outputs/notebook82076030c6 (1).ipynb` |
| Read the unexecuted 'fixed' source | `Food_101_Nutrition_Fixed.ipynb` |
| Run the project end-to-end on Kaggle | `notebooks/kaggle_weight_macros.ipynb` |
| Just check the code works locally | `scripts/smoke_test.py` (no notebook needed) |