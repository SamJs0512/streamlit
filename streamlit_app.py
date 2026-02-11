import streamlit as st
import pickle
import numpy as np

# --------------------------
# PAGE CONFIG
# --------------------------
st.set_page_config(page_title="Fitness Class Predictor", layout="wide")

# --------------------------
# LOAD MODEL
# --------------------------
with open("fitness_classifier.pkl", "rb") as f:
    model = pickle.load(f)

# --------------------------
# SESSION STATE
# --------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

# --------------------------
# GLOBAL CSS (PARALLAX + STYLE)
# --------------------------
st.markdown("""
<style>
html {
    scroll-behavior: smooth;
}

body {
    margin: 0;
    background-color: #0d1b2a;
    color: white;
}

.hero {
    height: 100vh;
    background: linear-gradient(rgba(13,27,42,0.8), rgba(13,27,42,0.8)),
                url("https://images.unsplash.com/photo-1517836357463-d25dfeac3438") center/cover fixed;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

.section {
    height: 100vh;
    background: linear-gradient(rgba(13,27,42,0.85), rgba(13,27,42,0.85)),
                url("https://images.unsplash.com/photo-1558611848-73f7eb4001ab") center/cover fixed;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
}

h1 {
    font-size: 60px;
}

.result-section {
    height: 100vh;
    background: linear-gradient(rgba(13,27,42,0.9), rgba(13,27,42,0.9)),
                url("https://images.unsplash.com/photo-1517838277536-f5f99be501cd") center/cover fixed;
    display: flex;
    align-items: center;
    justify-content: center;
}
</style>
""", unsafe_allow_html=True)

# --------------------------
# HOME SECTION
# --------------------------
st.markdown('<div class="hero">', unsafe_allow_html=True)
st.markdown("<h1>🏋️ Predict Your Fitness Class</h1>", unsafe_allow_html=True)

if st.button("Start Now"):
    st.session_state.page = "predict"
    st.markdown("<script>window.location.href='#predict';</script>", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# --------------------------
# PREDICTION SECTION
# --------------------------
st.markdown('<div id="predict" class="section">', unsafe_allow_html=True)

st.header("Enter Your Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 10, 80)
    height = st.number_input("Height (cm)", 100, 220)
    weight = st.number_input("Weight (kg)", 30, 150)

with col2:
    grip = st.number_input("Grip Strength")
    situps = st.number_input("Sit-ups Count")
    broad_jump = st.number_input("Broad Jump (cm)")

if st.button("Predict"):
    features = np.array([[age, height, weight, grip, situps, broad_jump]])
    prediction = model.predict(features)

    st.session_state.result = prediction[0]
    st.markdown("<script>window.location.href='#result';</script>", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# --------------------------
# RESULT SECTION
# --------------------------
if "result" in st.session_state:
    st.markdown('<div id="result" class="result-section">', unsafe_allow_html=True)
    st.markdown(f"<h1>Your Fitness Class: {st.session_state.result}</h1>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
