import streamlit as st

def initialize_workout_state():
    if "workout_logs" not in st.session_state:
        st.session_state["workout_logs"] = []

def log_exercise_set(exercise, weight, reps, rpe):
    st.session_state["workout_logs"].append({
        "exercise": exercise,
        "weight": weight,
        "reps": reps,
        "rpe": rpe
    })