import streamlit as st

def initialize_nutrition_state():
    if "food_logs" not in st.session_state:
        st.session_state["food_logs"] = []

def log_meal(item):
    st.session_state["food_logs"].append(item)

def get_daily_summary():
    logs = st.session_state.get("food_logs", [])
    summary = {"calories": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0}
    for item in logs:
        summary["calories"] += float(item.get("calories", 0))
        summary["protein"] += float(item.get("protein", 0))
        summary["carbs"] += float(item.get("carbs", 0))
        summary["fat"] += float(item.get("fat", 0))
    return summary