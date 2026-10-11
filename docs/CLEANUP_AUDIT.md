# Cleanup Audit (proposal only - nothing was deleted)

This document is a **read-only audit**. It reports what exists, which files are
duplicates, and what could be cleaned up. **No file has been deleted, moved, or
renamed as part of this documentation task**, and nothing here is an
instruction to delete. Deleting anything requires your explicit approval.

---

## 1. Inventory snapshot (at audit time)

| Group | Files | Size |
|---|---|---|
| Project root (code/config/docs/artifacts) | 7 | 6.67 MB |
| `app/` | 2 | 0.03 MB |
| `backups/` | 1 | 0.03 MB |
| `configs/` | 3 | ~0 MB |
| `data/` | 1 | 0.03 MB |
| `docs/` | 3 | 0.01 MB |
| `model_outputs/` | 5 | 135.86 MB |
| `models/` | 3 | 59.15 MB |
| `notebooks/` | 1 | 0.01 MB |
| `scripts/` | 11 | 0.04 MB |
| `src/` | 62 | 0.40 MB |
| `tests/` | 44 | 0.46 MB |
| `.pytest_cache/` | 5 | 0.01 MB |
| **Total** | **148** | **202.71 MB** |

After this documentation task, the tree grew by the files listed at the end of
`FILE_BY_FILE_REFERENCE.md` (7 new docs + `backups/README.md`).

The size is dominated by trained models:

- `model_outputs/` ≈ 136 MB: `food101_nutrition_final.keras` (52.5 MB),
  `best_food101_finetuned.keras` (52.5 MB), `best_food101_stage1.keras`
  (29.9 MB), plus a 952 KB notebook and 1.5 KB class names.
- `models/` ≈ 59 MB: `classifier.keras` (52.5 MB) + 6.6 MB detector + 1.5 KB
  labels. **This is the runtime set - do not delete.**

---

## 2. Duplicate analysis (SHA-256 verified)

| # | Files (A vs B) | Verdict |
|---|---|---|
| 1 | `Food_101_Nutrition_Fixed.ipynb` (33,487 B) vs `backups/Food_101_Nutrition_Fixed.backup.ipynb` (33,487 B) | **Byte-identical.** Both unexecuted. `backups/` is the historical safety copy; the root copy is redundant for running, keep or remove by choice |
| 2 | `nutrition.csv` (28,108 B) vs `data/nutrition.csv` (28,108 B) | **Byte-identical.** Runtime uses `data/nutrition.csv`; the root copy is redundant |
| 3 | `model_outputs/food101_nutrition_final.keras` (55,087,618 B) vs `models/classifier/classifier.keras` (55,087,618 B) | **Byte-identical.** Runtime uses `models/`; `model_outputs/` copy is provenance (the original notebook output) - keep for the record |
| 4 | `model_outputs/class_names.json` (1,487 B) vs `models/classifier/class_names.json` (1,487 B) | **Byte-identical.** Same pattern as above |
| 5 | `yolov8n.pt` (6,929,007 B) vs `models/food_detector/food_detector.pt` (6,929,007 B) | **Byte-identical.** Runtime uses `models/`; root copy is redundant |
| 6 | `model_outputs/best_food101_finetuned.keras` (55,087,618 B) vs `model_outputs/food101_nutrition_final.keras` (55,087,618 B) | **Not** identical bytes, same size. Two serializations of the stage-2 fine-tuned model (best-epoch checkpoint vs final save). Both are provenance |
| 7 | `model_outputs/best_food101_stage1.keras` (31,332,507 B) | Unique: the stage-1 (head-only) checkpoint. Not used at runtime; provenance only |

---

## 3. Generated/cache files - always safe to regenerate

| Path | Files | Size | Note |
|---|---|---|---|
| `__pycache__/` (all) | 60 files | ~0.70 MB | Python bytecode, recreated on import |
| `.pytest_cache/` | 5 files | ~0.01 MB | Pytest last-run cache, recreated on `pytest` |

These are the only files that are unambiguously safe to delete and that
``pip`/`pytest` will recreate on their own.

---

## 4. Candidate cleanups (each needs YOUR approval)

These are *recommendations*, listed from most to least clearly beneficial.
Approximate space freed is noted. **None have been executed.**

| Proposal | Freed | Risk | Recommendation |
|---|---|---|---|
| Delete all `__pycache__/` + `.pytest_cache/` | ~0.71 MB | None (regenerated) | Safe to do at any time |
| Delete root `nutrition.csv` (keep `data/nutrition.csv`) | 28 KB | None (byte-identical duplicate) | Safe |
| Delete root `yolov8n.pt` (keep `models/food_detector/food_detector.pt`) | 6.6 MB | None (byte-identical duplicate) | Safe |
| Delete root `Food_101_Nutrition_Fixed.ipynb` (keep `backups/` copy) | 33 KB | None (byte-identical) | Safe, but keep the `backups/` copy |
| Archive `model_outputs/best_food101_stage1.keras` off-project | 29.9 MB | Low (not used at runtime) | Only if you need the ~30 MB |
| Archive `model_outputs/best_food101_finetuned.keras` off-project | 52.5 MB | Medium if cited as "best" checkpoint | Keep if you want full training provenance |
| Archive nothing from `models/` | — | — | `models/` is the runtime set; always keep |

If ALL of the above were applied, the project would keep its exact runtime
behaviour while freeing roughly **89 MB** (≈44% of its current 202.71 MB), and
the byte-identical duplicates would be gone.

---

## 5. Files that must NOT be deleted

- `models/classifier/classifier.keras`, `models/classifier/class_names.json`
- `models/food_detector/food_detector.pt`
- `data/nutrition.csv`
- `src/` (the whole package)
- `app/gradio_app.py`
- `scripts/` (the 5 CLI tools)
- `configs/` (`default.json`, `multi_dish.json`, `food_densities.json`)
- `tests/`
- `docs/`
- `model_outputs/notebook82076030c6 (1).ipynb` (training provenance - metrics traceability)
- `notebooks/`
- `requirements*.txt`, `.gitignore`, `README.md`

---

## 6. What was changed by this documentation task

Only two kinds of changes were made, and **no dataset, notebook, or trained
artifact was touched**:

1. `README.md` was rewritten (previous content preserved at
   `backups/README.md`).
2. Seven new guides were added under `docs/`
   (`PROJECT_GUIDE.md`, `FILE_BY_FILE_REFERENCE.md`,
   `MODELS_DATASETS_AND_METRICS.md`, `NOTEBOOK_GUIDE.md`,
   `RUN_AND_TROUBLESHOOT.md`, `CLEANUP_AUDIT.md`, `FRIEND_HANDOVER.md`).

**Wait for explicit approval before acting on any item in section 4.**