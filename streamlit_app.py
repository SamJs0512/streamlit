import streamlit as st
import pandas as pd
import joblib
import base64
import time

st.set_page_config(layout="wide", page_title="Fitness AI Predictor")

# =============================
# LOAD MODEL
# =============================
# Ensure 'fitness_classifier.pkl' is in the same directory
bundle = joblib.load("fitness_classifier.pkl")
model = bundle["model"]
feature_columns = bundle["columns"]

# =============================
# IMAGE ENCODER
# =============================
def get_base64(file):
    try:
        with open(file, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return "" # Fallback if images are missing

gym_bg = get_base64("assets/gym-bg.jpg")
dumbbell = get_base64("assets/dumbbell.png")

# =============================
# CSS STYLING (Unchanged Styles)
# =============================
st.markdown(f"""
<style>
#MainMenu {{visibility:hidden;}}
footer {{visibility:hidden;}}
html {{ scroll-behavior:smooth; }}
body {{ margin:0; padding:0;}}

/* HERO SECTION */
.hero {{
    height:100vh;
    background:
      linear-gradient(135deg, rgba(0,0,0,0.6), rgba(0,0,0,0.9)),
      url("data:image/jpg;base64,{gym_bg}");
    background-size:cover;
    background-attachment:fixed;
    display:flex;
    justify-content:center;
    align-items:center;
    flex-direction:column;
    color:white;
    text-align:center;
    position:relative;
}}

.hero h1 {{
    font-size:80px;
    font-weight:900;
    animation:fadeIn 2s ease-in-out;
}}

.hero p {{
    font-size:24px;
    opacity:0.9;
    animation:fadeIn 3s ease-in-out;
}}

.dumbbell {{
    position:absolute;
    width:160px;
    right:10%;
    top:35%;
    animation:float 5s ease-in-out infinite;
}}

@keyframes float {{
  0% {{transform:translateY(0px);}}
  50% {{transform:translateY(-25px);}}
  100% {{transform:translateY(0px);}}
}}

/* FORM SECTION */
.form-section {{
    min-height:100vh;
    background:
      linear-gradient(to bottom, rgba(0,0,0,0.85), rgba(0,0,0,0.95)),
      url("data:image/jpg;base64,{gym_bg}");
    background-size:cover;
    background-attachment:fixed;
    display:flex;
    justify-content:center;
    align-items:center;
    padding:50px 0;
}}

.glass {{
    backdrop-filter:blur(25px);
    background:rgba(255,255,255,0.08);
    border-radius:30px;
    padding:50px 70px;
    width:80%;
    max-width:1000px;
    box-shadow:0 10px 60px rgba(0,0,0,0.7);
    color:white;
    display:flex;
    flex-direction:column;
    gap:25px;
}}

.glass h3 {{
    font-size:36px;
    margin-bottom:20px;
    text-align:center;
}}

.stButton>button {{
    background: #ff8c00;
    color:white;
    font-size:20px;
    font-weight:700;
    border-radius:30px;
    padding:12px 35px;
    transition:0.3s;
    width: 100%;
}}
.stButton>button:hover {{
    background:#ffa533;
}}

@keyframes fadeIn {{
    from {{opacity:0;}}
    to {{opacity:1;}}
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
    <p>AI-powered elite performance analytics</p>
</div>
""", unsafe_allow_html=True)

# =============================
# FORM
# =============================
st.markdown('<div id="form" class="form-section">', unsafe_allow_html=True)
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.markdown("### Enter Your Body Metrics & Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("🏃 **Physical Profile**")
    age = st.slider("Age", 10, 90, 25)
    gender = st.selectbox("Gender", ["M", "F"])
    height = st.number_input("Height (cm)", 130.0, 220.0, 175.0)
    weight = st.number_input("Weight (kg)", 30.0, 160.0, 70.0)

with col2:
    st.write("🩺 **Health Markers**")
    bodyfat = st.number_input("Body Fat (%)", 1.0, 50.0, 18.0)
    systolic = st.number_input("Systolic BP", 80, 200, 120)
    diastolic = st.number_input("Diastolic BP", 40, 120, 80)
    grip = st.number_input("Grip Strength (kg)", 5.0, 80.0, 45.0)

with col3:
    st.write("💪 **Performance Tests**")
    flexibility = st.number_input("Sit & Bend (cm)", -30.0, 50.0, 15.0)
    situps = st.number_input("Sit-ups Count", 0, 100, 40)
    jump = st.number_input("Broad Jump (cm)", 0, 400, 200)

st.markdown("<br>", unsafe_allow_html=True)
submit = st.button("Predict Fitness Level")
st.markdown("</div></div>", unsafe_allow_html=True)

# =============================
# PREDICTION LOGIC
# =============================
if submit:
    try:
        # 1. Create Initial DataFrame
        df_input = pd.DataFrame([{
            "age": age,
            "gender": gender,
            "height_cm": height,
            "weight_kg": weight,
            "body fat_%": bodyfat,
            "diastolic": diastolic,
            "systolic": systolic,
            "gripForce": grip,
            "sit and bend forward_cm": flexibility,
            "sit-ups counts": situps,
            "broad jump_cm": jump
        }])

        # 2. Gender Encoding 
        df_input["gender_M"] = 1 if gender == "M" else 0
        df_input = df_input.drop(columns=["gender"])

        # 3. Feature Engineering
        df_input["BMI"] = df_input["weight_kg"] / ((df_input["height_cm"] / 100) ** 2)
        df_input["BP_ratio"] = df_input["systolic"] / (df_input["diastolic"] + 0.1)
        df_input["age_grip"] = df_input["age"] * df_input["gripForce"]
        df_input["weight_height_ratio"] = df_input["weight_kg"] / df_input["height_cm"]
        df_input["strength_weight"] = df_input["gripForce"] / df_input["weight_kg"]
        df_input["bodyfat_BMI"] = df_input["body fat_%"] * df_input["BMI"]

        # 4. Reindex
        df_input = df_input.reindex(columns=feature_columns, fill_value=0)

        # 5. Prediction
        prediction = model.predict(df_input)[0]

        # 6. Result Display
        st.balloons()
        st.success(f"🏁 Predicted Fitness Class: **{prediction}**")
        
        # Fixed formatting for the info box
        st.markdown("""
        ### Class Guide:
        - **A**: Excellent performance
        - **B**: Good performance
        - **C**: Average performance
        - **D**: Needs Improvement
        """)

    except Exception as e:
        st.error(f"Version Mismatch Error: {e}")
        st.info("To fix this: Re-run your training script to generate a new .pkl file and upload it to GitHub.")