import streamlit as st
import time

def initialize_fasting_state():
    if "water_ml" not in st.session_state:
        st.session_state["water_ml"] = 0
    if "fasting_state" not in st.session_state:
        st.session_state["fasting_state"] = {"is_fasting": False, "start_time": None, "target_hours": 16}

def log_water(amount_ml):
    st.session_state["water_ml"] += amount_ml

def render_fasting_widget():
    st.subheader("⏱️ Intermittent Fasting Timer")
    state = st.session_state["fasting_state"]
    
    if not state["is_fasting"]:
        target = st.selectbox("Target Fasting Window", [12, 14, 16, 18, 20, 24], index=2)
        if st.button("🚀 Start Fast"):
            state["is_fasting"] = True
            state["start_time"] = time.time()
            state["target_hours"] = target
            st.rerun()
    else:
        elapsed = time.time() - state["start_time"]
        elapsed_hours = elapsed / 3600.0
        progress = min(1.0, elapsed_hours / state["target_hours"])
        
        st.progress(progress, text=f"Elapsed: {elapsed_hours:.2f} hrs / Goal: {state['target_hours']} hrs")
        if st.button("🛑 End Fast"):
            state["is_fasting"] = False
            st.success(f"Completed fast of {elapsed_hours:.2f} hours!")
            st.rerun()