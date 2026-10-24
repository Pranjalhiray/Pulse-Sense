import joblib
import pandas as pd
import streamlit as st


st.set_page_config(page_title="PulseSense · Heart Health", page_icon="♥", layout="wide", initial_sidebar_state="expanded")


@st.cache_resource
def load_artifacts():
    return joblib.load("KNN_heart.pkl"), joblib.load("scaler.pkl"), joblib.load("columns.pkl")


model, scaler, expected_columns = load_artifacts()

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root { --ink:#eef4ff; --muted:#9caac4; --line:rgba(164,186,228,.16); }
    .stApp { background:radial-gradient(circle at 15% 0%,rgba(91,73,192,.25),transparent 32rem),radial-gradient(circle at 92% 10%,rgba(255,78,114,.17),transparent 27rem),#080b14; color:var(--ink); font-family:'DM Sans',sans-serif; }
    .block-container { max-width:1320px; padding:2.8rem 3.4rem 4rem; }
    [data-testid="stSidebar"] { background:rgba(9,12,24,.72); border-right:1px solid var(--line); }
    [data-testid="stSidebar"] > div:first-child { padding:2.4rem 1.4rem; }
    h1,h2,h3,h4 { font-family:'Manrope',sans-serif !important; letter-spacing:-.04em; color:var(--ink) !important; }
    h1 { font-size:clamp(2.5rem,5vw,4.5rem) !important; line-height:.98 !important; margin:.35rem 0 1rem !important; }
    h2 { font-size:1.45rem !important; }
    p,label,.stCaption { color:var(--muted) !important; }
    .eyebrow { color:#ff8798; font-size:.74rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; }
    .hero-copy { max-width:690px; font-size:1.08rem; line-height:1.65; color:#aebbd4; }
    .hero-icon { display:inline-flex; align-items:center; justify-content:center; width:3.2rem; height:3.2rem; margin-bottom:1rem; border:1px solid rgba(255,129,149,.35); border-radius:1.1rem; background:linear-gradient(145deg,rgba(255,95,118,.25),rgba(139,123,255,.18)); color:#ff899b; font-size:1.45rem; box-shadow:0 0 36px rgba(255,95,118,.18); }
    .glass-card { padding:1.35rem 1.55rem; border:1px solid var(--line); border-radius:1.35rem; background:linear-gradient(135deg,rgba(32,39,64,.66),rgba(14,18,32,.58)); box-shadow:0 20px 50px rgba(0,0,0,.2),inset 0 1px rgba(255,255,255,.045); backdrop-filter:blur(18px); }
    .mini-card { min-height:106px; padding:1rem 1.1rem; border:1px solid var(--line); border-radius:1rem; background:rgba(24,30,51,.56); }
    .mini-label { color:var(--muted); font-size:.72rem; letter-spacing:.08em; text-transform:uppercase; }
    .mini-value { color:var(--ink); font-family:'Manrope'; font-size:1.25rem; font-weight:700; margin-top:.3rem; }
    .section-label { display:flex; align-items:center; gap:.55rem; margin:.3rem 0 1rem; color:#dfe8fc; font-family:'Manrope'; font-size:1rem; font-weight:700; }
    .section-label span { display:inline-flex; align-items:center; justify-content:center; width:1.6rem; height:1.6rem; border-radius:.55rem; background:rgba(139,123,255,.18); color:#ad9fff; font-size:.76rem; }
    .stTextInput input,.stNumberInput input,div[data-baseweb="select"] > div,.stSlider > div { background-color:rgba(8,12,24,.62) !important; border-color:var(--line) !important; color:var(--ink) !important; }
    .stButton > button { width:100%; min-height:3.2rem; border:0; border-radius:.9rem; background:linear-gradient(100deg,#ff5f76,#a26cff); color:white; font-family:'Manrope'; font-weight:800; box-shadow:0 12px 28px rgba(255,95,118,.23); transition:transform .2s,filter .2s; }
    .stButton > button:hover { filter:brightness(1.12); transform:translateY(-2px); color:white; }
    .result-high,.result-low { padding:1.2rem 1.35rem; border-radius:1.1rem; margin-top:1rem; }
    .result-high { border:1px solid rgba(255,95,118,.35); background:linear-gradient(130deg,rgba(255,75,102,.18),rgba(55,26,54,.4)); }
    .result-low { border:1px solid rgba(73,211,166,.3); background:linear-gradient(130deg,rgba(53,187,153,.16),rgba(22,47,54,.4)); }
    .result-title { font-family:'Manrope'; font-size:1.15rem; font-weight:800; color:var(--ink); }
    .result-copy { color:#afbdd7; font-size:.9rem; line-height:1.5; margin-top:.35rem; }
    .disclaimer { color:#687792; font-size:.75rem; line-height:1.5; }
    </style>
    """, unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("<div class='eyebrow'>PulseSense</div>", unsafe_allow_html=True)
    st.markdown("### ♥ Heart Health")
    st.markdown("A calm, private way to explore cardiovascular risk signals from a few clinical measurements.")
    st.divider()
    st.markdown("**How it works**")
    st.markdown("1. Enter the available health markers.\n2. Run the model assessment.\n3. Use the result as a conversation starter with a clinician.")
    st.divider()
    st.markdown("<div class='disclaimer'>This tool is for educational screening only. It does not diagnose disease or replace professional medical advice.</div>", unsafe_allow_html=True)

st.markdown("<div class='hero-icon'>♥</div><div class='eyebrow'>Personal health snapshot</div>", unsafe_allow_html=True)
st.title("Understand your heart health signals.")
st.markdown("<div class='hero-copy'>A thoughtfully designed screening experience powered by your trained machine-learning model. Share a few details and get a clear, instant signal.</div>", unsafe_allow_html=True)
st.write("")

with st.form("health_profile"):
    left, right = st.columns([1.05, 1], gap="large")
    with left:
        st.markdown("<div class='section-label'><span>01</span> About you</div>", unsafe_allow_html=True)
        age = st.slider("Age", 18, 100, 40, help="Your age in years.")
        sex = st.selectbox("Sex", ["M", "F"], format_func=lambda value: "Male" if value == "M" else "Female")
        resting_bp = st.number_input("Resting blood pressure", min_value=80, max_value=200, value=120, step=1, help="Systolic resting blood pressure in mm Hg.")
        cholesterol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200, step=1, help="Serum cholesterol in mg/dl.")
    with right:
        st.markdown("<div class='section-label'><span>02</span> Heart signals</div>", unsafe_allow_html=True)
        chest_pain = st.selectbox("Chest pain type", ["ATA", "NAP", "ASY", "TA"], format_func=lambda value: {"ATA":"Atypical angina","NAP":"Non-anginal pain","ASY":"Asymptomatic","TA":"Typical angina"}[value])
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy(LVH)"], format_func=lambda value: "LV hypertrophy" if value.startswith("Left") else value)
        max_hr = st.slider("Maximum heart rate", 60, 220, 150, help="Maximum heart rate achieved during exercise.")
        exercise_angina = st.selectbox("Exercise-induced angina", ["Y", "N"], format_func=lambda value: "Yes" if value == "Y" else "No")
    st.write("")
    extra_left, extra_right = st.columns(2, gap="large")
    with extra_left:
        fasting_bs = st.selectbox("Fasting blood sugar above 120 mg/dl", [0, 1], format_func=lambda value: "Yes" if value == 1 else "No")
    with extra_right:
        oldpeak = st.number_input("ST depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    st_slope = st.selectbox("Slope of peak exercise ST segment", ["UP", "FLAT", "DOWN"], format_func=lambda value: {"UP":"Upsloping","FLAT":"Flat","DOWN":"Downsloping"}[value])
    submitted = st.form_submit_button("Run my heart health check  →")

if submitted:
    raw_input = {"Age":age,"RestingBP":resting_bp,"FastingBS":fasting_bs,"Cholesterol":cholesterol,"MaxHR":max_hr,"OldPeak":oldpeak,
                 f"Sex_{sex}":1,f"ChestPainType_{chest_pain}":1,f"RestingECG_{resting_ecg}":1,f"ExerciseAngina_{exercise_angina}":1,f"ST_Slope_{st_slope}":1}
    input_df = pd.DataFrame([raw_input])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]
    prediction_input = scaler.transform(input_df)
    prediction = model.predict(prediction_input)[0]
    probability = float(model.predict_proba(prediction_input)[0][1]) if hasattr(model, "predict_proba") else None
    st.markdown("### Your screening snapshot")
    metric_a, metric_b, metric_c = st.columns(3)
    with metric_a:
        st.markdown(f"<div class='mini-card'><div class='mini-label'>Age</div><div class='mini-value'>{age} years</div></div>", unsafe_allow_html=True)
    with metric_b:
        st.markdown(f"<div class='mini-card'><div class='mini-label'>Blood pressure</div><div class='mini-value'>{resting_bp} mm Hg</div></div>", unsafe_allow_html=True)
    with metric_c:
        score = f"{probability:.0%}" if probability is not None else ("Elevated" if prediction == 1 else "Lower")
        st.markdown(f"<div class='mini-card'><div class='mini-label'>Model signal</div><div class='mini-value'>{score}</div></div>", unsafe_allow_html=True)
    if prediction == 1:
        st.markdown("<div class='result-high'><div class='result-title'>⚠ Elevated risk signal detected</div><div class='result-copy'>The model found patterns associated with higher cardiovascular risk. Please discuss these results and your measurements with a qualified healthcare professional.</div></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='result-low'><div class='result-title'>✦ Lower risk signal detected</div><div class='result-copy'>The model did not identify a higher-risk pattern in this snapshot. Continue routine check-ups and healthy habits.</div></div>", unsafe_allow_html=True)
    st.caption("A model prediction is not a diagnosis. If you have urgent symptoms, seek medical care immediately.")
