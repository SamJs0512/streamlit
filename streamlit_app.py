import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sklearn

st.write("Sklearn version on Streamlit:", sklearn.__version__)

# ==============================
# Load Model
# ==============================
bundle = joblib.load("fitness_classifier.pkl")
model = bundle["model"]
feature_columns = bundle["columns"]

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
    grip = st.number_input("Grip Strength", 5, 80, 40)

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
        # Create dataframe with SAME column names as training
        df_input = pd.DataFrame([{
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
            "gender": gender
        }])

        # Encode gender (same method as training)
        df_input = pd.get_dummies(df_input, columns=["gender"], drop_first=True)

        # ==============================
        # Feature Engineering (MUST MATCH TRAINING)
        # ==============================
        df_input["bmi"] = df_input["weight_kg"] / ((df_input["height_cm"] / 100) ** 2)
        df_input["bp_ratio"] = df_input["systolic"] / (df_input["diastolic"] + 0.1)
        df_input["age_grip"] = df_input["age"] * df_input["gripforce"]
        df_input["strength_weight"] = df_input["gripforce"] / df_input["weight_kg"]
        df_input["bodyfat_bmi"] = df_input["body_fat_pct"] * df_input["bmi"]

        # Align columns exactly as training
        df_input = df_input.reindex(columns=feature_columns, fill_value=0)

        # Predict
        prediction = model.predict(df_input)[0]
        probabilities = model.predict_proba(df_input)[0]

        st.success(f"🏁 Predicted Fitness Class: **{prediction}**")

        st.write("### 📊 Prediction Confidence")
        for cls, prob in zip(model.classes_, probabilities):
            st.write(f"{cls}: {prob:.2%}")

        st.info("""
        **A** – Excellent  
        **B** – Good  
        **C** – Average  
        **D** – Needs Improvement
        """)

    except Exception as e:
        st.error(f"Prediction Error: {e}")
