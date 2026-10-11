# Limitations and Honesty Notes

This project deliberately refuses to invent values. The following are
**unverified locally** because the data, GPU, or optional packages are absent in
the development sandbox. They must be validated where those resources exist.

## What is verified (executed locally)

- The full pytest suite passes (`python -m pytest tests -q`).
- A real tiny classifier trains, saves (`classifier.keras`), reloads
  (`compile=False`), and predicts — see `scripts/smoke_test.py`.
- The nutrition lookup/scaling, weight-trust ordering, detection guards, the
  pipeline, reporting, and all CLI argument handling run on synthetic inputs.

## What is NOT verified locally

| Area | Why | What is required |
|---|---|---|
| Real Food-101 accuracy | no dataset, no GPU | attach Food-101 + train, then run `evaluate.py` |
| Real YOLO food detection | bundled `models/food_detector/food_detector.pt` loads and detects real food (verified, 113 classes — not COCO) | rerun on your own photos to sanity-check |
| Nutrition5k mass regression | ~23.5 GB mirror not attached | attach the mirror; regressor trains/loads |
| Depth-based weight | needs RGB-D + calibration; disabled by default | depth dataset + calibration; keep off until validated |
| Gradio UI launch | gradio installed and launched locally (HTTP 200 at `http://127.0.0.1:7860`); UI flows are unit-tested | public hosting (e.g. `--share` or Hugging Face Spaces) still unverified |
| Kaggle notebook run | no `/kaggle/input` locally | run `notebooks/kaggle_weight_macros.ipynb` on Kaggle |
| ViT fusion | `keras-hub` not installed | install keras-hub; path raises a clear error otherwise |

## Multi-dish totals

- Per-dish weights must be supplied per dish (CLI `--dish-grams`, the UI
  **Grams (g)** column, or `apply_weights`). A single `--measured` weight is used
  only when exactly one dish is present.
- A **plate total** is shown only when every *included* dish has both a usable
  weight and nutrition data. Otherwise it is reported as `unavailable` with a
  reason; the project never presents the sum of reference-quantity macros as a
  real plate total.

## Weight semantics

- `measured` — user-supplied grams; the only value treated as ground truth.
- `estimated` — Nutrition5k regression, only accepted inside the configured
  reliable range (`weight.min_reliable_grams` .. `max_reliable_grams`).
- `experimental` — depth volume × approximate density; off by default and
  labelled low-confidence.
- `unavailable` — nothing reliable; **no grams are shown**, and macros fall back
  to a clearly-labelled reference quantity.

## Detection

Ultralytics pretrained models on the default download path are COCO and contain
**no food classes**, so the project ships an explicit food detector at
`models/food_detector/food_detector.pt` (a 113-class Nutrition5k-style model).
It is loaded with `verify_class_names=False` because its class list differs from
Food-101; the detector finds food *boxes* and the classifier assigns the
Food-101 label. **Verified locally:** the weights load and detect real food.
Detected boxes describe *where* food is, never *how much* it weighs.

## Classification confidence

A softmax confidence is not a correctness guarantee. The UI and reports present
top-k predictions and never claim accuracy that was not measured.
