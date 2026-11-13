# AI Food Macro Tracker

An end-to-end computer vision application that estimates meal macronutrients and mass directly from a standard RGB image.

## Pipeline Architecture
1. **Object Detection**: YOLO framework extracts bounding boxes around food items.
2. **Hybrid Identification**: Crops are classified using an EfficientNet-V2 + Vision Transformer (ViT) ensemble.
3. **Volumetric Engine**: MiDaS constructs a 3D relative depth map to deduce volume and approximate mass using specific food density factors.
4. **Nutrition UI**: A modern Streamlit dashboard maps predicted mass to the USDA API (or local database fallback).

## Setup Instructions

1. **Install Dependencies**
   Ensure you have Python 3.8+ installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Custom Weights (Optional)**
   Place your pre-trained GFSD YOLO weights in the `weights/` folder as `gfsd_yolo_weights.pt`.
   *Note: If custom weights are missing, the application includes an automatic fallback to `yolov8n.pt` to ensure it launches gracefully.*

3. **Configure Gemini**
   In PowerShell, set the `GEMINI_API_KEY` environment variable to a valid Google Gemini API key:
   ```powershell
   $env:GEMINI_API_KEY = "your-api-key"
   ```

4. **Run the Application**
   ```bash
   streamlit run app.py
   ```
