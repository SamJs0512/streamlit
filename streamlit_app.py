import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==============================
# Load Model Bundle
# ==============================
try:
    bundle = joblib.load("fitness_classifier.pkl")
    model = bundle["model"]
    feature_columns = bundle["columns"]
except Exception as e:
    st.error(f"Could not load model file. Please run train.py first. Error: {e}")
    st.stop()

st.set_page_config(page_title="Fitness Classifier", layout="wide")

st.title("🏋️ Fitness Level Classification App")
st.write("Enter your body and performance metrics to predict your fitness class.")

# ==============================
# User Input Section
# ==============================
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 10, 90, 25)
    gender = st.selectbox("Gender", ["M", "F"])
    height = st.number_input("Height (cm)", 130, 220, 170)
    weight = st.number_input("Weight (kg)", 30, 160, 70)

with col2:
    bodyfat = st.number_input("Body Fat (%)", 1.0, 50.0, 18.0)
    systolic = st.number_input("Systolic BP", 80, 200, 120)
    diastolic = st.number_input("Diastolic BP", 40, 120, 80)
    grip = st.number_input("Grip Strength", 5.0, 80.0, 40.0)

with col3:
    flex = st.number_input("Sit & Bend Forward (cm)", -30.0, 50.0, 15.0)
    situps = st.number_input("Sit-ups Count", 0, 100, 40)
    jump = st.number_input("Broad Jump (cm)", 0, 400, 200)

predict_btn = st.button("Predict Fitness Level")

# ==============================
# Prediction Logic
# ==============================
if predict_btn:
    try:
        # 1. Create initial dataframe with core metrics
        # Use the SAME names as defined in train.py before dummies
        input_data = {
            "age": age,
            "height_cm": height,
            "weight_kg": weight,
            "body_fat_pct": bodyfat,
            "diastolic": diastolic,
            "systolic": systolic,
            "gripforce": grip,
            "sit_and_bend_forward_cm": flex,
            "sit-ups_counts": situps,
            "broad_jump_cm": jump,
        }
        
        df_input = pd.DataFrame([input_data])

        # 2. Manual Encoding (Robust for single rows)
        # Based on training logic, F is the baseline (0) and M is the feature (1)
        df_input["gender_M"] = 1 if gender == "M" else 0

        # 3. Feature Engineering (EXACT match to training)
        df_input["bmi"] = df_input["weight_kg"] / ((df_input["height_cm"] / 100) ** 2)
        df_input["bp_ratio"] = df_input["systolic"] / (df_input["diastolic"] + 0.1)
        df_input["age_grip"] = df_input["age"] * df_input["gripforce"]
        df_input["strength_weight"] = df_input["gripforce"] / df_input["weight_kg"]
        df_input["bodyfat_bmi"] = df_input["body_fat_pct"] * df_input["bmi"]

        # 4. Align columns exactly as training (Order matters for Random Forest)
        df_input = df_input.reindex(columns=feature_columns, fill_value=0)

        # 5. Predict
        prediction = model.predict(df_input)[0]
        probabilities = model.predict_proba(df_input)[0]

        # Display Results
        st.divider()
        st.success(f"🏁 Predicted Fitness Class: **{prediction}**")

        st.write("### 📊 Prediction Confidence")
        cols = st.columns(len(model.classes_))
        for i, (cls, prob) in enumerate(zip(model.classes_, probabilities)):
            cols[i].metric(label=f"Class {cls}", value=f"{prob:.1%}")

        st.info("""
        **Class Key:** **A** – Excellent | **B** – Good | **C** – Average | **D** – Needs Improvement
        """)

    except Exception as e:
        st.error(f"Prediction Error: {e}")