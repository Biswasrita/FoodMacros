
import os
import cv2
import numpy as np
import streamlit as st

from PIL import Image
from dotenv import load_dotenv

load_dotenv()

from models.detector import FoodDetector
from models.llm_classifier import LLMFoodClassifier


# -----------------------------
# STREAMLIT CONFIGURATION
# -----------------------------

st.set_page_config(
    layout="wide",
    page_title="AI Food Macro Tracker"
)

st.title("🍽️ AI Food Macro Tracker")

st.markdown(
    "Upload a meal image to detect food items "
    "and estimate their nutritional values."
)


# -----------------------------
# LOAD MODELS
# -----------------------------

if not os.getenv("GEMINI_API_KEY"):
    st.error("Please configure your GEMINI_API_KEY.")
    st.stop()


@st.cache_resource
def load_models():

    classifier = LLMFoodClassifier()
    detector = FoodDetector()

    return detector, classifier


try:
    detector, classifier = load_models()

except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()


# -----------------------------
# OVERLAP CALCULATION
# -----------------------------

def calculate_overlap(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    if intersection == 0:
        return 0, 0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    iou = intersection / union if union > 0 else 0

    smaller_area = min(area1, area2)

    ios = intersection / smaller_area if smaller_area > 0 else 0

    return iou, ios


# -----------------------------
# REMOVE DUPLICATE CROPS
# -----------------------------

def filter_redundant_crops(boxes, crops):

    if not boxes:
        return [], []

    indices = sorted(
        range(len(boxes)),
        key=lambda i: boxes[i]["conf"],
        reverse=True
    )

    keep = []

    for i in indices:

        duplicate = False

        for j in keep:

            iou, ios = calculate_overlap(
                boxes[i]["box"],
                boxes[j]["box"]
            )

            if iou > 0.45 or ios > 0.70:
                duplicate = True
                break

        if not duplicate:
            keep.append(i)

    return (
        [boxes[i] for i in keep],
        [crops[i] for i in keep]
    )


# -----------------------------
# IMAGE UPLOAD
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload a food image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    with st.spinner("Detecting food items..."):

        raw_boxes, raw_crops = detector.detect(image)

        boxes, crops = filter_redundant_crops(
            raw_boxes,
            raw_crops
        )


    # -----------------------------
    # DISPLAY DETECTION
    # -----------------------------

    img_boxes = img_np.copy()

    for item in boxes:

        x1, y1, x2, y2 = item["box"]

        cv2.rectangle(
            img_boxes,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )


    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Detected Food Items")

        st.image(
            img_boxes,
            use_container_width=True
        )


    # -----------------------------
    # FOOD CLASSIFICATION
    # -----------------------------

    with col2:

        st.subheader("Nutritional Breakdown")

        total_cal = 0
        total_pro = 0
        total_fat = 0
        total_carb = 0

        if not crops:
            st.warning("No food items detected.")

        for i, crop in enumerate(crops):

            try:

                with st.spinner(f"Analysing food item {i + 1}..."):
                    item = classifier.predict(crop)

            except Exception as error:

                st.error(
                    f"Analysis failed for item {i + 1}: {error}"
                )
                continue


            label = item["label"]
            confidence = item["confidence"]
            portion = item["portion_unit"]

            cal = item["calories"]
            pro = item["protein"]
            fat = item["fat"]
            carb = item["carbs"]
            fiber = item["fiber"]

            # Add to total nutrition
            total_cal += cal
            total_pro += pro
            total_fat += fat
            total_carb += carb


            with st.expander(
                f"Item {i + 1}: {label.title()} - {cal:.0f} kcal"
            ):

                st.write(f"**Portion:** {portion}")

                st.write(
                    f"**Confidence:** {confidence * 100:.1f}%"
                )

                st.write(f"**Calories:** {cal:.1f} kcal")
                st.write(f"**Protein:** {pro:.1f} g")
                st.write(f"**Fat:** {fat:.1f} g")
                st.write(f"**Carbohydrates:** {carb:.1f} g")
                st.write(f"**Fiber:** {fiber:.1f} g")


        # -----------------------------
        # TOTAL NUTRITION
        # -----------------------------

        st.divider()

        st.subheader("Total Nutrition")

        st.metric(
            "Total Calories",
            f"{total_cal:.1f} kcal"
        )

        st.progress(
            min(total_pro / 150, 1.0),
            text=f"Protein: {total_pro:.1f} g"
        )

        st.progress(
            min(total_fat / 100, 1.0),
            text=f"Fat: {total_fat:.1f} g"
        )

        st.progress(
            min(total_carb / 300, 1.0),
            text=f"Carbohydrates: {total_carb:.1f} g"
        )

        st.caption(
            "Nutrition values are AI-generated estimates, "
            "not laboratory measurements."
        )
