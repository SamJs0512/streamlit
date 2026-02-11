import streamlit as st
import pandas as pd
import joblib
import base64

st.set_page_config(layout="wide")

# =============================
# Load model
# =============================
bundle = joblib.load("fitness_classifier.pkl")
model = bundle["model"]
feature_columns = bundle["columns"]

# =============================
# Helper to encode images
# =============================
def get_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()

gym_bg = get_base64("assets/gym-bg.jpg")
dumbbell = get_base64("assets/dumbbell.png")

# =============================
# PREMIUM CSS + ANIMATIONS
# =============================
st.markdown(f"""
<style>

/* Smooth scroll */
html {{
  scroll-behavior: smooth;
}}

/* Remove Streamlit UI */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}

/* Global */
section {{
    scroll-snap-align: start;
}}

body {{
    scroll-snap-type: y mandatory;
}}

/* HERO SECTION */
.hero {{
    height: 100vh;
    background:
        linear-gradient(to bottom, rgba(0,0,0,0.7), rgba(0,0,0,0.9)),
        url("data:image/jpg;base64,{gym_bg}");
    background-size: cover;
    background-attachment: fixed;
    background-position: center;
    display: flex;
    justify-content: center;
    align-items: center;
    flex-direction: column;
    color: white;
    position: relative;
    text-align: center;
}}

.hero h1 {{
    font-size: 70px;
    animation: fadeIn 2s ease-in-out;
}}

.hero p {{
    font-size: 22px;
    opacity: 0.8;
    animation: fadeIn 3s ease-in-out;
}}

.next-btn {{
    padding: 15px 45px;
    background: white;
    color: black;
    border-radius: 40px;
    font-weight: bold;
    text-decoration: none;
    margin-top: 40px;
    transition: 0.3s;
}}

.next-btn:hover {{
    background: #f39c12;
    color: white;
}}

/* Dumbbell scroll slide */
.dumbbell {{
    position: absolute;
    width: 130px;
    right: 10%;
    top: 25%;
    transition: transform 0.5s ease-out;
}}

.hero:hover .dumbbell {{
    transform: translateX(-100px) rotate(20deg);
}}

/* FORM SECTION */
.form-section {{
    height: 100vh;
    background: linear-gradient(to bottom, #0f0f0f, #1a1a1a);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: white;
}}

.glass {{
    backdrop-filter: blur(12px);
    background: rgba(255,255,255,0.05);
    border-radius: 20px;
    padding: 40px;
    width: 60%;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
    animation: fadeInUp 1.5s ease-in-out;
}}

/* RESULT SECTION */
.result-section {{
    height: 100vh;
    background:
        linear-gradient(to top, rgba(0,0,0,0.9), rgba(0,0,0,0.7)),
        url("data:image/jpg;base64,{gym_bg}");
    background-size: cover;
    background-attachment: fixed;
    display: flex;
    justify-content: center;
    align-items: center;
    color: white;
    font-size: 80px;
    font-weight: bold;
    animation: fadeIn 2s ease-in-out;
}}

/* Animations */
@keyframes fadeIn {{
    from {{opacity: 0;}}
    to {{opacity: 1;}}
}}

@keyframes fadeInUp {{
    from {{
        opacity: 0;
        transform: translateY(50px);
    }}
    to {{
        opacity: 1;
        transform: translateY(0);
    }}
}}

</style>
""", unsafe_allow_html=True)

# =============================
# HERO
# =============================
st.markdown(f"""
<div class="hero">
    <img src="data:image/png;base64,{dumbbell}" class="dumbbell">
    <h1>Predict Your Fitness Class</h1>
    <p>Scroll down and discover your true performance level</p>
    <a href="#form" class="next-btn">Start Now ↓</a>
</div>
""", unsafe_allow_html=True)

# =============================
# FORM SECTION
# =============================
st.markdown('<div id="form" class="form-section">', unsafe_allow_html=True)
st.markdown('<div class="glass">', unsafe_allow_html=True)

st.markdown("### Enter Your Body Metrics")

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

predict = st.button("Predict Fitness Level")

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# =============================
# PREDICTION
# =============================
if predict:

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

    st.markdown(f"""
    <div class="result-section">
        🏆 CLASS {prediction}
    </div>
    """, unsafe_allow_html=True)
