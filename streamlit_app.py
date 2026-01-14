import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# 1. Load model bundle
# -----------------------------
bundle = joblib.load("fitness_classifier_compact.pkl")
model = bundle["model"]
feature_columns = bundle["columns"]

st.title("🏋️ Gym Member Fitness Level Prediction (A–D)")
st.write("Enter your measurements and exercise performance:")

# -----------------------------
# 2. User inputs
# -----------------------------
age = st.slider("Age", 10, 90, 25)
gender = st.selectbox("Gender", ["M", "F"])
height = st.number_input("Height (cm)", 130, 220, 170)
weight = st.number_input("Weight (kg)", 30, 160, 70)
bodyfat = st.number_input("Body fat (%)", 1.0, 50.0, 18.0)
diastolic = st.number_input("Diastolic BP", 40, 120, 80)
systolic = st.number_input("Systolic BP", 80, 200, 120)
grip = st.number_input("Grip Force", 5, 70, 35)

# -----------------------------
# 3. Prepare dataframe
# -----------------------------
df_input = pd.DataFrame([{
    "age": age,
    "gender": gender,
    "height_cm": height,
    "weight_kg": weight,
    "body_fat_pct": bodyfat,
    "diastolic": diastolic,
    "systolic": systolic,
    "gripForce": grip
}])

# one-hot encode gender
df_input = pd.get_dummies(df_input, columns=["gender"], drop_first=True)

# align columns to training features
df_input = df_input.reindex(columns=feature_columns, fill_value=0)

# -----------------------------
# 4. Predict
# -----------------------------
if st.button("Predict Fitness Level"):
    pred = model.predict(df_input)[0]
    st.success(f"🏁 Predicted Fitness Level: **{pred}**")
