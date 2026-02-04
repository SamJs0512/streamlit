import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Fitness Level Predictor",
    page_icon="💪",
    layout="centered"
)

# ===============================
# Load model
# ===============================
bundle = joblib.load("fitness_classifier.pkl")
model = bundle["model"]
feature_columns = bundle["columns"]

# ===============================
# UI
# ===============================
st.markdown(
    "<h1 style='text-align:center;'>💪 Gym Fitness Level Predictor</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;color:gray;'>Predict fitness class A–D</p>",
    unsafe_allow_html=True
)

st.divider()

with st.form("form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 10, 90, 25)
        height = st.number_input("Height (cm)", 130, 220, 170)
        weight = st.number_input("Weight (kg)", 30, 160, 70)
        bodyfat = st.number_input("Body Fat (%)", 1.0, 50.0, 18.0)

    with col2:
        gender = st.selectbox("Gender", ["M", "F"])
        systolic = st.number_input("Systolic BP", 80, 200, 120)
        diastolic = st.number_input("Diastolic BP", 40, 120, 80)
        grip = st.number_input("Grip Strength", 5, 70, 35)

    submit = st.form_submit_button("Predict")

# ===============================
# Prediction
# ===============================
if submit:
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

    prediction = model.predict(df_input)[0]

    st.success(f"🏁 Predicted Fitness Class: **{prediction}**")

    st.info("""
    **A** – Excellent  
    **B** – Good  
    **C** – Average  
    **D** – Needs Improvement
    """)
