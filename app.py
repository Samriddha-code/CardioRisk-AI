import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

# =========================
# Page Setup
# =========================
st.set_page_config(
    page_title="CardioRisk AI",
    page_icon="🫀",
    layout="wide"
)

# =========================
# Load Model
# =========================
@st.cache_resource
def load_model():
    xgb_model = XGBClassifier()
    xgb_model.load_model("heart_disease_model.json")
    return xgb_model

@st.cache_resource
def load_threshold():
    return joblib.load("decision_threshold.pkl")

model = load_model()
threshold = load_threshold()

# =========================
# Dark Professional CSS
# =========================
st.markdown("""
<style>
.stApp {
    background: #0F172A;
    color: #E5E7EB;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1250px;
}

section[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #334155;
}

h1, h2, h3, h4, h5, h6 {
    color: #F8FAFC !important;
}

p, label, span {
    color: #E5E7EB !important;
}

[data-testid="stMetric"] {
    background-color: #111827;
    border: 1px solid #334155;
    padding: 16px;
    border-radius: 14px;
}

[data-testid="stMetricLabel"] {
    color: #94A3B8 !important;
}

[data-testid="stMetricValue"] {
    color: #F8FAFC !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #111827;
    border: 1px solid #334155;
    border-radius: 16px;
}

.stNumberInput input {
    background-color: #1E293B;
    color: #F8FAFC;
    border: 1px solid #334155;
}

.stSelectbox div[data-baseweb="select"] > div {
    background-color: #1E293B;
    color: #F8FAFC;
    border-color: #334155;
}

.hero {
    background: linear-gradient(135deg, #1E293B, #020617);
    padding: 36px 42px;
    border-radius: 20px;
    border: 1px solid #334155;
    margin-bottom: 22px;
}

.hero-title {
    font-size: 44px;
    font-weight: 800;
    color: #F8FAFC !important;
    letter-spacing: -0.5px;
}

.hero-subtitle {
    font-size: 17px;
    color: #CBD5E1 !important;
    margin-top: 8px;
    max-width: 850px;
}

.section-title {
    font-size: 24px;
    font-weight: 700;
    color: #F8FAFC !important;
    margin-top: 18px;
    margin-bottom: 12px;
}

.result-high {
    border: 1px solid #EF4444;
    background-color: #450A0A;
    color: #FCA5A5 !important;
    padding: 28px;
    border-radius: 18px;
    text-align: center;
    font-size: 30px;
    font-weight: 800;
    margin-top: 20px;
}

.result-low {
    border: 1px solid #22C55E;
    background-color: #052E16;
    color: #86EFAC !important;
    padding: 28px;
    border-radius: 18px;
    text-align: center;
    font-size: 30px;
    font-weight: 800;
    margin-top: 20px;
}

.small-note {
    color: #94A3B8 !important;
    font-size: 14px;
}

.footer {
    text-align: center;
    color: #94A3B8 !important;
    margin-top: 35px;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Sidebar
# =========================
st.sidebar.title("CardioRisk AI")
st.sidebar.markdown("---")
st.sidebar.write("**Model:** Recall-Tuned XGBoost")
st.sidebar.write(f"**Decision Threshold:** {threshold}")
st.sidebar.write("**Recall:** 78.09%")
st.sidebar.write("**F1 Score:** 73.63%")
st.sidebar.write("**ROC-AUC:** 80.02%")
st.sidebar.markdown("---")
st.sidebar.warning("Educational project only. Not for real medical diagnosis.")

# =========================
# Header
# =========================
st.markdown("""
<div class="hero">
    <div class="hero-title">CardioRisk AI</div>
    <div class="hero-subtitle">
        AI-powered cardiovascular disease risk assessment using a recall-optimized XGBoost model.
    </div>
</div>
""", unsafe_allow_html=True)

st.info(
    "Disclaimer: This application is for educational and research purposes only. "
    "It is not a medical diagnostic tool and should not replace professional medical advice."
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Model", "XGBoost")
m2.metric("Recall", "78.09%")
m3.metric("ROC-AUC", "80.02%")
m4.metric("Threshold", f"{threshold}")

st.divider()

# =========================
# Main Layout
# =========================
left, right = st.columns([1.2, 0.8])

# =========================
# Inputs
# =========================
with left:
    st.markdown('<div class="section-title">Patient Information</div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.subheader("Basic Details")
        c1, c2 = st.columns(2)

        with c1:
            age_years = st.number_input("Age", min_value=18.0, max_value=100.0, value=50.0, step=1.0)
            height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=165.0, step=1.0)

        with c2:
            gender = st.selectbox(
                "Gender",
                options=[1, 2],
                format_func=lambda x: "Female" if x == 1 else "Male"
            )
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=250.0, value=70.0, step=1.0)

    with st.container(border=True):
        st.subheader("Vital Signs")
        c1, c2 = st.columns(2)

        with c1:
            ap_hi = st.number_input("Systolic Blood Pressure", min_value=50, max_value=250, value=120, step=1)

        with c2:
            ap_lo = st.number_input("Diastolic Blood Pressure", min_value=30, max_value=200, value=80, step=1)

    with st.container(border=True):
        st.subheader("Laboratory Values")
        c1, c2 = st.columns(2)

        with c1:
            cholesterol = st.selectbox(
                "Cholesterol Level",
                options=[1, 2, 3],
                format_func=lambda x: {
                    1: "Normal",
                    2: "Above Normal",
                    3: "Well Above Normal"
                }[x]
            )

        with c2:
            gluc = st.selectbox(
                "Glucose Level",
                options=[1, 2, 3],
                format_func=lambda x: {
                    1: "Normal",
                    2: "Above Normal",
                    3: "Well Above Normal"
                }[x]
            )

    with st.container(border=True):
        st.subheader("Lifestyle")
        c1, c2, c3 = st.columns(3)

        with c1:
            smoke = 1 if st.toggle("Smoking") else 0

        with c2:
            alco = 1 if st.toggle("Alcohol Consumption") else 0

        with c3:
            active = 1 if st.toggle("Physically Active", value=True) else 0

# =========================
# Feature Engineering
# =========================
BMI = weight / ((height / 100) ** 2)
pulse_pressure = ap_hi - ap_lo
MAP = ap_lo + (pulse_pressure / 3)

input_data = pd.DataFrame([{
    "gender": gender,
    "height": height,
    "weight": weight,
    "ap_hi": ap_hi,
    "ap_lo": ap_lo,
    "cholesterol": cholesterol,
    "gluc": gluc,
    "smoke": smoke,
    "alco": alco,
    "active": active,
    "age_years": age_years,
    "BMI": BMI,
    "pulse_pressure": pulse_pressure,
    "MAP": MAP
}])

# =========================
# Health Summary
# =========================
with right:
    st.markdown('<div class="section-title">Health Summary</div>', unsafe_allow_html=True)

    with st.container(border=True):
        h1, h2, h3 = st.columns(3)
        h1.metric("BMI", f"{BMI:.2f}")
        h2.metric("Pulse Pressure", f"{pulse_pressure:.1f} mmHg")
        h3.metric("MAP", f"{MAP:.2f} mmHg")

        if BMI < 18.5:
            bmi_category = "Underweight"
        elif BMI < 25:
            bmi_category = "Normal"
        elif BMI < 30:
            bmi_category = "Overweight"
        else:
            bmi_category = "Obese"

        st.write(f"**BMI Category:** {bmi_category}")

        if ap_hi >= 140 or ap_lo >= 90:
            st.warning("Blood pressure is elevated.")
        else:
            st.success("Blood pressure is within normal range.")

    with st.container(border=True):
        st.subheader("Health Assessment")

        st.write(f"**BMI Category:** {bmi_category}")

        if ap_hi >= 140 or ap_lo >= 90:
            st.warning("Elevated blood pressure detected.")
        else:
            st.success("Blood pressure is within the normal range.")

        if cholesterol == 1:
            st.success("Cholesterol: Normal")
        elif cholesterol == 2:
            st.warning("Cholesterol: Above Normal")
        else:
            st.error("Cholesterol: Well Above Normal")

        if gluc == 1:
            st.success("Glucose: Normal")
        elif gluc == 2:
            st.warning("Glucose: Above Normal")
        else:
            st.error("Glucose: Well Above Normal")

# =========================
# Prediction
# =========================
st.divider()

if st.button("Predict Cardiovascular Risk", use_container_width=True):
    probability = model.predict_proba(input_data)[:, 1][0]
    prediction = 1 if probability >= threshold else 0

    st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)

    r1, r2, r3 = st.columns(3)
    r1.metric("Risk Probability", f"{probability * 100:.2f}%")
    r2.metric("Threshold Used", f"{threshold * 100:.0f}%")

    if probability >= 0.75:
        confidence = "High"
    elif probability >= threshold:
        confidence = "Moderate"
    else:
        confidence = "Low"

    r3.metric("Risk Confidence", confidence)

    if prediction == 1:
        st.markdown('<div class="result-high">High Risk of Heart Disease</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-low">Low Risk of Heart Disease</div>', unsafe_allow_html=True)

    st.progress(float(probability))

    st.markdown('<div class="section-title">Detected Risk Factors</div>', unsafe_allow_html=True)

    risk_factors = []

    if ap_hi >= 140:
        risk_factors.append("High systolic blood pressure")
    if ap_lo >= 90:
        risk_factors.append("High diastolic blood pressure")
    if cholesterol == 2:
        risk_factors.append("Above normal cholesterol")
    if cholesterol == 3:
        risk_factors.append("Well above normal cholesterol")
    if gluc == 2:
        risk_factors.append("Above normal glucose")
    if gluc == 3:
        risk_factors.append("Well above normal glucose")
    if BMI >= 30:
        risk_factors.append("Obesity")
    elif BMI >= 25:
        risk_factors.append("Overweight")
    if active == 0:
        risk_factors.append("Physical inactivity")
    if smoke == 1:
        risk_factors.append("Smoking")
    if alco == 1:
        risk_factors.append("Alcohol consumption")
    if age_years >= 55:
        risk_factors.append("Higher age group")

    if risk_factors:
        with st.container(border=True):
            for factor in risk_factors:
                st.write(f"• {factor}")
    else:
        st.success("No major visible risk factors detected.")

    st.markdown('<div class="section-title">Risk Probability Visualization</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(7, 3))
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#111827")

    ax.barh(["Low Risk", "High Risk"], [1 - probability, probability], color=["#22C55E", "#EF4444"])
    ax.axvline(threshold, color="#FACC15", linestyle="--", linewidth=2, label="Decision Threshold")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability", color="white")
    ax.set_title("Predicted Risk Distribution", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#111827", edgecolor="#334155", labelcolor="white")

    st.pyplot(fig)

    with st.expander("View Data Sent to Model"):
        st.dataframe(input_data)

# =========================
# Footer
# =========================
st.markdown("""
<div class="footer">
    Built with Python, Streamlit, XGBoost and Scikit-Learn.
</div>
""", unsafe_allow_html=True)
