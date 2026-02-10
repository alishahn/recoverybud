import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ✅ You can change this later after you decide your threshold policy
THRESHOLD = 0.45

MODEL_PATH = "recovery_rf_model.pkl"

st.set_page_config(page_title="Recovery Readiness", layout="centered")
st.title("🏃 Recovery Readiness Advisor")
st.caption("Predicts underperformance risk using your saved preprocess pipeline + Random Forest.")

@st.cache_resource
def load_model():
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

model = load_model()

# ============================================================
# ✅ These MUST match preprocess.feature_names_in_ + categories_
# ============================================================
EXPECTED_FEATURES = list(model.named_steps["preprocess"].feature_names_in_)

# Categories from your screenshot (exact casing!)
GENDER_OPTS = ["F", "M", "Other"]
SMOKING_OPTS = ["Current", "Former", "Never"]
PREV_INTENSITY_OPTS = ["High", "Low", "Medium", "Unknown"]
ACTIVITY_TYPE_OPTS = ["Basketball", "Cycling", "Dancing", "HIIT", "Running",
                      "Swimming", "Tennis", "Walking", "Weight Training", "Yoga"]
INTENSITY_OPTS = ["High", "Low", "Medium"]
ACTIVITY_ENV_OPTS = ["Indoor", "Outdoor"]
BMI_CAT_OPTS = ["Normal", "Obese", "Overweight", "Underweight"]

# ============================================================
# Helper: compute BMI category exactly like training
# bins=[0, 18.5, 25, 30, 100], labels=["Underweight","Normal","Overweight","Obese"]
# ============================================================
def compute_bmi_category(bmi_value: float) -> str:
    if bmi_value < 18.5:
        return "Underweight"
    elif bmi_value < 25:
        return "Normal"
    elif bmi_value < 30:
        return "Overweight"
    else:
        return "Obese"

# =========================
# Sidebar inputs
# =========================
st.sidebar.header("Pre-workout inputs")

sleep_3day_avg = st.sidebar.slider("sleep_3day_avg (hours)", 0.0, 12.0, 7.0, 0.1)
hydration_3day_avg = st.sidebar.slider("hydration_3day_avg", 0.0, 10.0, 6.0, 0.1)
stress_3day_avg = st.sidebar.slider("stress_3day_avg", 0.0, 10.0, 4.0, 0.1)
cumulative_sleep_deficit = st.sidebar.slider("cumulative_sleep_deficit (hours)", 0.0, 30.0, 2.0, 0.1)
days_since_rest = st.sidebar.slider("days_since_rest", 0, 21, 2, 1)
recovery_score = st.sidebar.slider("recovery_score", 0.0, 100.0, 70.0, 1.0)

# IMPORTANT: must be categorical in your training
previous_intensity = st.sidebar.selectbox("previous_intensity", PREV_INTENSITY_OPTS)

activity_type = st.sidebar.selectbox("activity_type", ACTIVITY_TYPE_OPTS)
intensity = st.sidebar.selectbox("intensity", INTENSITY_OPTS)
activity_environment = st.sidebar.selectbox("activity_environment", ACTIVITY_ENV_OPTS)

st.sidebar.subheader("Profile / health inputs")

age = st.sidebar.slider("age", 10, 90, 18, 1)
gender = st.sidebar.selectbox("gender", GENDER_OPTS)
height_cm = st.sidebar.slider("height_cm", 120.0, 210.0, 156.0, 0.5)

# ✅ Add weight + hydration_level so we can compute hydration_per_kg
weight_kg = st.sidebar.slider("weight_kg", 35.0, 150.0, 55.0, 0.5)
hydration_level = st.sidebar.slider("hydration_level (liters/day)", 1.0, 4.0, 2.2, 0.1)

bmi = st.sidebar.number_input("bmi", min_value=10.0, max_value=60.0, value=20.5, step=0.1)

fitness_level_normalized = st.sidebar.slider("fitness_level_normalized", 0.0, 1.0, 0.45, 0.01)
resting_heart_rate = st.sidebar.slider("resting_heart_rate", 30, 120, 60, 1)

blood_pressure_systolic = st.sidebar.slider("blood_pressure_systolic", 80, 200, 115, 1)
blood_pressure_diastolic = st.sidebar.slider("blood_pressure_diastolic", 40, 130, 75, 1)

smoking_status = st.sidebar.selectbox("smoking_status", SMOKING_OPTS)

# These are numeric 0/1 in your feature list
user_reported_condition = st.sidebar.selectbox("user_reported_condition (0=no/ 1=yes)", [0, 1])
reported_hypertension = st.sidebar.selectbox("reported_hypertension (0=no/ 1=yes)", [0, 1])
reported_diabetes = st.sidebar.selectbox("reported_diabetes (0=no/ 1=yes)", [0, 1])
reported_asthma = st.sidebar.selectbox("reported_asthma (0=no/ 1=yes)", [0, 1])

daily_steps = st.sidebar.number_input("daily_steps", min_value=0, max_value=50000, value=8000, step=500)

# ============================================================
# ✅ Derived features (must match training)
# ============================================================
# hydration_per_kg was used in training → derive from liters/day and weight
hydration_per_kg = float(hydration_level) / float(weight_kg)

# bmi_category was used in training → derive from bmi
bmi_category = compute_bmi_category(float(bmi))

# Engineered features your preprocess expects
# Based on your training: sleep_gap = 8 - sleep_3day_avg
sleep_gap = 8.0 - float(sleep_3day_avg)

# Based on your training: rest_pressure = cumulative_sleep_deficit * days_since_rest
rest_pressure = float(cumulative_sleep_deficit) * float(days_since_rest)

# =========================
# Main UI: show derived values
# =========================
st.subheader("Derived inputs (computed)")
st.write(f"**hydration_per_kg:** {hydration_per_kg:.4f} L/kg  (from {hydration_level:.1f} L/day ÷ {weight_kg:.1f} kg)")
st.write(f"**bmi_category:** {bmi_category}  (from BMI={float(bmi):.1f})")

# ============================================================
# Build raw input row with EXACT expected columns
# ============================================================
X_raw = pd.DataFrame([{
    "age": age,
    "gender": gender,
    "height_cm": height_cm,
    "bmi": bmi,
    "fitness_level_normalized": fitness_level_normalized,
    "resting_heart_rate": resting_heart_rate,
    "blood_pressure_systolic": blood_pressure_systolic,
    "blood_pressure_diastolic": blood_pressure_diastolic,
    "smoking_status": smoking_status,
    "user_reported_condition": user_reported_condition,
    "reported_hypertension": reported_hypertension,
    "reported_diabetes": reported_diabetes,
    "reported_asthma": reported_asthma,
    "daily_steps": daily_steps,
    "sleep_3day_avg": sleep_3day_avg,
    "hydration_3day_avg": hydration_3day_avg,
    "stress_3day_avg": stress_3day_avg,
    "cumulative_sleep_deficit": cumulative_sleep_deficit,
    "days_since_rest": days_since_rest,
    "recovery_score": recovery_score,
    "hydration_per_kg": hydration_per_kg,
    "previous_intensity": previous_intensity,
    "activity_type": activity_type,
    "intensity": intensity,
    "activity_environment": activity_environment,
    "bmi_category": bmi_category,
    "sleep_gap": sleep_gap,
    "rest_pressure": rest_pressure,
}])

# Ensure exact column ordering (important for some pipelines)
X_raw = X_raw.reindex(columns=EXPECTED_FEATURES)

st.subheader("Result")

try:
    # Model is a Pipeline, so pass X_raw directly
    p_risk = float(model.predict_proba(X_raw)[0, 1])
except Exception as e:
    st.error("Prediction failed (feature mismatch or datatype mismatch).")
    st.code(str(e))
    st.info(
        "Debug checklist:\n"
        "1) X_raw columns must match preprocess.feature_names_in_\n"
        "2) Categorical values must match training categories exactly (case-sensitive)\n"
        "3) user_reported_condition must be 0/1\n"
        "4) previous_intensity must be one of High/Low/Medium/Unknown\n"
        "5) activity_type/intensity/activity_environment must match your training categories"
    )
    with st.expander("Show X_raw"):
        st.dataframe(X_raw, use_container_width=True)
    st.stop()

risk_pct = p_risk * 100
recommendation = "REST / LIGHT SESSION" if p_risk >= THRESHOLD else "MODERATE or PUSH"
emoji = "🛑" if p_risk >= THRESHOLD else "✅"

st.metric("Risk of underperformance", f"{risk_pct:.1f}%")
st.progress(min(max(p_risk, 0.0), 1.0))
st.markdown(f"### Recommendation: {emoji} **{recommendation}**")
st.caption(f"Threshold = {THRESHOLD:.2f}  |  p_risk = {p_risk:.3f}")

with st.expander("Show input features (X_raw)"):
    st.dataframe(X_raw, use_container_width=True)