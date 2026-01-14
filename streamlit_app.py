import streamlit as st
import pandas as pd
import joblib

# load bundle
bundle = joblib.load("fitness_classifier_compact.pkl")
model = bundle["model"]
feature_columns = bundle["columns"]

st.title("Body Performance Classifier")

age = st.number_input("Age", 10, 100)
gender = st.selectbox("Gender", ["M", "F"])
height = st.number_input("Height (cm)", 100, 220)
weight = st.number_input("Weight (kg)", 30, 200)
bodyfat = st.number_input("Body fat (%)", 1.0, 60.0)
diastolic = st.number_input("Diastolic BP", 40, 120)
systolic = st.number_input("Systolic BP", 80, 200)
grip = st.number_input("Grip Force", 5, 80)

# build row
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

# align columns used during training
df_input = df_input.reindex(columns=feature_columns, fill_value=0)

if st.button("Predict"):
    pred = model.predict(df_input)[0]
    st.success(f"Predicted body performance class: {pred}")
