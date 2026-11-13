import streamlit as st
from PIL import Image
import numpy as np
import cv2
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Append current directory so modular imports work smoothly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.detector import FoodDetector
from models.llm_classifier import LLMFoodClassifier
from utils.nutrition_lookup import get_nutrition

st.set_page_config(layout="wide", page_title="AI Food Macro Tracker")

@st.cache_resource
def load_models():
    # Model caching avoids reloading weights on every UI interaction
    detector = FoodDetector()
    
    # HARDCODE YOUR GEMINI API KEY HERE
    api_key = (os.environ.get("api_key"))
    
    classifier = LLMFoodClassifier(api_key=api_key)
    return detector, classifier
import numpy as np

def calculate_overlap(box1, box2):
    """Calculates Intersection over Union (IoU) and Intersection over Smaller Box (IoS)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0, 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    iou = inter_area / float(union_area) if union_area > 0 else 0.0
    ios = inter_area / float(min(box1_area, box2_area))

    return iou, ios

def filter_redundant_crops(boxes, crops, iou_thresh=0.45, ios_thresh=0.70):
    """Filters out duplicate or highly overlapping bounding boxes to prevent double-counting calories."""
    if not boxes:
        return [], []

    # Sort boxes by confidence score (descending)
    sorted_indices = sorted(range(len(boxes)), key=lambda k: boxes[k]['conf'], reverse=True)
    keep_indices = []

    for i in sorted_indices:
        box_i = boxes[i]['box']
        redundant = False
        
        for j in keep_indices:
            box_j = boxes[j]['box']
            iou, ios = calculate_overlap(box_i, box_j)
            
            # If IoU > threshold OR box_i is > 70% inside an existing higher-confidence box
            if iou > iou_thresh or ios > ios_thresh:
                redundant = True
                break
                
        if not redundant:
            keep_indices.append(i)

    filtered_boxes = [boxes[k] for k in keep_indices]
    filtered_crops = [crops[k] for k in keep_indices]
    return filtered_boxes, filtered_crops

st.title("🍽️ AI Food Macro Tracker")
st.markdown("Combines **YOLO** and **Gemini Vision** to detect and calculate caloric density from an RGB image.")

detector, classifier = load_models()

uploaded_file = st.file_uploader("Upload an RGB Meal Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    
    with st.spinner("Running AI Pipeline..."):
        # 1. Detection
        raw_boxes, raw_crops = detector.detect(image)
        # 2. Deduplicate overlapping / redundant crops
        boxes, crops = filter_redundant_crops(raw_boxes, raw_crops)
    
    # Adjusted to 2 columns since Depth Map is removed
    col1, col2 = st.columns(2)
    
    # Overlay bounding boxes
    img_boxes = img_np.copy()
    for b in boxes:
        x1, y1, x2, y2 = b['box']
        cv2.rectangle(img_boxes, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
    with col1:
        st.image(img_boxes, caption="YOLOv8 Detection", use_container_width=True)
        
    with col2:
      st.subheader("Nutritional Breakdown (LLM Portion Estimate)")
      total_cal, total_pro, total_fat, total_carb = 0.0, 0.0, 0.0, 0.0

      if not crops:
        st.warning(
            "No food items were detected, so the food classifier was not run."
        )

      for i, crop in enumerate(crops):
        try:
          # 2. LLM Vision Classification & Direct Macro Estimation
          item = classifier.predict(crop)
        except Exception as error:
          st.error(f"Food classification failed: {error}")
          st.stop()

        label = item["label"]
        conf = item["confidence"]
        cal = item["calories"]
        pro = item["protein"]
        fat = item["fat"]
        carb = item["carbs"]
        fiber = item["fiber"]

        total_cal += cal
        total_pro += pro
        total_fat += fat
        total_carb += carb

        with st.expander(
            f"Item {i+1}: {label.title()} ({conf*100:.1f}% conf) — {cal:.0f}"
            " kcal"
        ):
          st.write(
              f"**Protein:** {pro:.1f}g | **Fat:** {fat:.1f}g | **Carbs:**"
              f" {carb:.1f}g | **Fiber:** {fiber:.1f}g"
          )

      st.write("---")
      st.metric("Total Calories", f"{total_cal:.1f} kcal")

      st.write("**Total Macronutrients**")
      st.progress(min(1.0, total_pro / 150.0), text=f"Protein: {total_pro:.1f}g")
      st.progress(min(1.0, total_fat / 100.0), text=f"Fat: {total_fat:.1f}g")
      st.progress(min(1.0, total_carb / 300.0), text=f"Carbs: {total_carb:.1f}g")