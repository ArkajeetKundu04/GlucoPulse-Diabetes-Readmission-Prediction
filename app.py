import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(
    page_title="GlucoPulse",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ----------------------------------
# SAFE Dark UI CSS (NO fragile selectors)
# ----------------------------------
st.markdown("""
<style>
/* App background */
.stApp {
    background-color: #0e1117;
    color: #ffffff;
}

/* Headings */
h1, h2, h3, h4 {
    color: #58a6ff;
}

/* Labels */
label {
    color: #ffffff !important;
    font-weight: 500;
}

/* Number input container */
div[data-baseweb="input"] > div {
    background-color: #161b22;
    border-radius: 10px;
    border: 1px solid #30363d;
}

/* Input text */
input {
    color: white !important;
    background-color: #161b22 !important;
}

/* Predict button */
.stButton > button {
    background-color: #238636;
    color: white;
    border-radius: 14px;
    padding: 0.7em 1.4em;
    font-size: 17px;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    background-color: #2ea043;
    transform: scale(1.05);
}

/* Progress bar */
.stProgress > div > div {
    background-color: #58a6ff;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------
# Title
# ----------------------------------
st.title("🩺 GlucoPulse")
st.subheader("Diabetes Readmission Prediction System")

# ----------------------------------
# Load Models
# ----------------------------------
rf = joblib.load("rf_model.pkl")
dt = joblib.load("dt_model.pkl")
xgb = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")
tf_model = tf.keras.models.load_model("tf_model.keras")

# ----------------------------------
# User Inputs
# ----------------------------------
st.header("📋 Enter Patient Details")

st.markdown("### 🧠 Clinical Features")

age = st.number_input("Age Group (encoded)", 0, 9, 5, 1)
time_in_hospital = st.number_input("Time in Hospital (days)", 1, 14, 3, 1)
num_lab_procedures = st.number_input("Number of Lab Procedures", 1, 150, 40, 1)
num_procedures = st.number_input("Number of Procedures", 0, 6, 1, 1)
num_medications = st.number_input("Number of Medications", 1, 80, 10, 1)
number_diagnoses = st.number_input("Number of Diagnoses", 1, 16, 5, 1)

st.markdown("### 🏥 Visit History")

number_outpatient = st.number_input("Outpatient Visits", 0, 10, 0, 1)
number_emergency = st.number_input("Emergency Visits", 0, 10, 0, 1)
number_inpatient = st.number_input("Inpatient Visits", 0, 10, 0, 1)

# ----------------------------------
# Feature Engineering
# ----------------------------------
num_medications_prescribed = num_medications
emergency_ratio = number_emergency / (time_in_hospital + 1)

# ----------------------------------
# Build Input DataFrame
# ----------------------------------
input_dict = {
    "age": age,
    "time_in_hospital": time_in_hospital,
    "num_lab_procedures": num_lab_procedures,
    "num_procedures": num_procedures,
    "num_medications": num_medications,
    "number_outpatient": number_outpatient,
    "number_emergency": number_emergency,
    "number_inpatient": number_inpatient,
    "number_diagnoses": number_diagnoses,
    "num_medications_prescribed": num_medications_prescribed,
    "emergency_ratio": emergency_ratio
}

input_df = pd.DataFrame([input_dict])

# Align columns
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[feature_columns]

# Scale
input_scaled = scaler.transform(input_df)

# ----------------------------------
# Prediction
# ----------------------------------
if st.button("🔍 Predict Readmission"):
    rf_prob = float(rf.predict_proba(input_scaled)[0][1])
    dt_prob = float(dt.predict_proba(input_scaled)[0][1])
    xgb_prob = float(xgb.predict_proba(input_scaled)[0][1])
    tf_prob = float(tf_model.predict(input_scaled)[0][0])

    avg_prob = np.mean([rf_prob, dt_prob, xgb_prob, tf_prob])

    st.subheader("🧾 Final Prediction")

    if avg_prob >= 0.5:
        st.error("🔴 Patient Likely to be Readmitted")
    else:
        st.success("🟢 Patient Unlikely to be Readmitted")

    st.markdown("### 📊 Model Confidence Scores")

    st.progress(int(rf_prob * 100))
    st.write(f"Random Forest: **{rf_prob:.2f}**")

    st.progress(int(dt_prob * 100))
    st.write(f"Decision Tree: **{dt_prob:.2f}**")

    st.progress(int(xgb_prob * 100))
    st.write(f"XGBoost: **{xgb_prob:.2f}**")

    st.progress(int(tf_prob * 100))
    st.write(f"Logistic Regression (TensorFlow): **{tf_prob:.2f}**")
