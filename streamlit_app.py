import streamlit as st
import pandas as pd
import joblib

bundle = joblib.load("fitness_classifier_compact.pkl")
model = bundle["model"]
feature_columns = bundle["columns"]

# Example input
age = st.slider("Age", 10, 90, 25)
gender = st.selectbox("Gender", ["M", "F"])
height = st.number_input("Height (cm)", 100, 220, 170)
weight = st.number_input("Weight (kg)", 30, 150, 70)
bodyfat = st.number_input("Body fat %", 1.0, 50.0, 18.0)
diastolic = st.number_input("Diastolic BP", 40, 120, 80)
systolic = st.number_input("Systolic BP", 80, 200, 120)
grip = st.number_input("Grip Force", 5, 70, 30)

# Prepare dataframe
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

df_input = pd.get_dummies(df_input, columns=["gender"], drop_first=True)
df_input = df_input.reindex(columns=feature_columns, fill_value=0)

if st.button("Predict"):
    pred = model.predict(df_input)[0]
    st.success(f"Predicted Fitness Level: {pred}")
