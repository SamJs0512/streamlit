import joblib
import streamlit as st
import pandas as pd

# Load trained model
model = joblib.load("bodyPerformance.csv")

st.title("Gym Member Fitness Level Prediction")

st.write("Predict fitness level (A–D) based on body measurements and exercise performance")

# User inputs
age = st.slider("Age", 10, 90, 25)

gender = st.selectbox("Gender", ["M", "F"])

height_cm = st.slider("Height (cm)", 130, 210, 170)
weight_kg = st.slider("Weight (kg)", 30, 160, 70)
bodyfat = st.slider("Body Fat %", 1.0, 50.0, 18.0)

diastolic = st.slider("Diastolic BP", 40, 120, 80)
systolic = st.slider("Systolic BP", 80, 200, 120)

grip = st.slider("Grip Force", 5, 70, 35)
sitbend = st.slider("Sit & Bend Forward (cm)", -30, 30, 5)
situps = st.slider("Sit Ups (count)", 0, 80, 30)
broadjump = st.slider("Broad Jump (cm)", 50, 300, 160)

if st.button("Predict Fitness Class"):

    # Create dataframe for input
    df_input = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "height_cm": [height_cm],
        "weight_kg": [weight_kg],
        "body fat_%": [bodyfat],
        "diastolic": [diastolic],
        "systolic": [systolic],
        "gripForce": [grip],
        "sit_and_bend_forward": [sitbend],
        "sit_ups": [situps],
        "broad_jump": [broadjump]
    })

    # One-hot encode gender
    df_input = pd.get_dummies(df_input, columns=["gender"])

    # Align with model training features
    df_input = df_input.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    # Predict
    pred_class = model.predict(df_input)[0]

    st.success(f"Predicted Fitness Level: **{pred_class}**")

# Optional background design
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)
