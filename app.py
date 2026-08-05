# ========== 1. IMPORTS ==========
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import csv
from datetime import datetime
from pathlib import Path

# ========== 2. PAGE CONFIG + STYLING ==========
st.set_page_config(
    page_title="HSC AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast CSS Design System
st.markdown("""
<style>
/* Import Modern Google Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Primary Button Styling */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0284c7 0%, #0d9488 100%) !important;
    color: #ffffff !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    padding: 12px 24px !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(13, 148, 136, 0.35) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}

div.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(13, 148, 136, 0.5) !important;
}

/* Secondary Button Styling */
div.stButton > button[kind="secondary"] {
    border-radius: 10px !important;
    font-weight: 500 !important;
}

/* Card Containers */
.feature-card {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.6) 0%, rgba(30, 41, 59, 0.6) 100%);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    backdrop-filter: blur(10px);
}

.how-it-works-box {
    background: #0f172a;
    border: 1px solid #334155;
    border-left: 5px solid #38bdf8;
    border-radius: 12px;
    padding: 20px;
    color: #f8fafc !important;
    margin-bottom: 20px;
}

.how-it-works-box h4 {
    color: #38bdf8 !important;
    margin-top: 0;
    margin-bottom: 12px;
    font-size: 1.1rem;
    font-weight: 600;
}

.how-it-works-box ol {
    margin-bottom: 0;
    padding-left: 20px;
    color: #e2e8f0 !important;
    line-height: 1.8;
}

.how-it-works-box li {
    color: #e2e8f0 !important;
}

/* Prediction Cards */
.hero-prediction-card {
    background: linear-gradient(135deg, rgba(2, 132, 199, 0.15) 0%, rgba(13, 148, 136, 0.15) 100%);
    border: 2px solid #0284c7;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
}

.hero-prediction-card h3 {
    color: #38bdf8 !important;
    margin: 0;
    font-size: 0.95rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.hero-prediction-card h2 {
    color: #ffffff !important;
    margin: 8px 0 0 0;
    font-size: 2rem;
    font-weight: 700;
}

/* Risk Badges */
.severity-badge {
    padding: 16px;
    border-radius: 12px;
    text-align: center;
    font-weight: 600;
}

.severity-high {
    background-color: rgba(225, 29, 72, 0.2);
    border: 1.5 solid #f43f5e;
    color: #fecdd3;
}

.severity-moderate {
    background-color: rgba(217, 119, 6, 0.2);
    border: 1.5px solid #f59e0b;
    color: #fef3c7;
}

.severity-low {
    background-color: rgba(16, 185, 129, 0.2);
    border: 1.5px solid #10b981;
    color: #d1fae5;
}

/* Metric Display Cards */
.stat-card {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}

.stat-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #38bdf8;
    line-height: 1;
    margin-bottom: 4px;
}

.stat-label {
    font-size: 0.85rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if "page" not in st.session_state: 
    st.session_state.page = "home"
if "admin_logged_in" not in st.session_state: 
    st.session_state.admin_logged_in = False
if "clear_counter" not in st.session_state: 
    st.session_state.clear_counter = 0
if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False
    st.session_state.prediction = ""
    st.session_state.top3_diseases = []
    st.session_state.top3_probs = []
    st.session_state.selected_symptoms = []
    st.session_state.username = "Guest"

def go_to_checker(): 
    st.session_state.page = "checker"

def go_to_home():
    st.session_state.page = "home"
    st.session_state.prediction_made = False

BASE_DIR = Path(__file__).resolve().parent

# ========== 3. LOAD MODEL & DATASETS ==========
@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / "best_model.pkl")

@st.cache_resource
def load_support_files():
    encoder = joblib.load(BASE_DIR / "label_encoder.pkl")
    feature_columns = joblib.load(BASE_DIR / "feature_columns.pkl")
    all_symptoms = list(feature_columns)
    diseases = list(encoder.classes_)

    disease_info = {d: {'description': "Description not available", 'precautions': ["Consult a healthcare professional."]} for d in diseases}
    try:
        desc_df = pd.read_csv(BASE_DIR / "symptoms_Africa20.csv")
        prec_df = pd.read_csv(BASE_DIR / "precaution_Africa20.csv")
        merged_df = pd.merge(desc_df, prec_df, on='Disease', how='outer')

        for idx in range(len(merged_df)):
            row = merged_df.iloc[idx]
            disease = str(row['Disease'])
            desc = str(row['Description']) if 'Description' in merged_df.columns and pd.notna(row['Description']) else "No description available."

            precautions = []
            for i in range(1, 5):
                col = f'Precaution_{i}'
                if col in merged_df.columns and pd.notna(row[col]):
                    val = str(row[col]).strip()
                    if val and val.lower() != 'nan':
                        precautions.append(val.title())

            disease_info[disease] = {'description': desc, 'precautions': precautions if precautions else ["Consult A Healthcare Professional."]}
    except Exception as e:
        st.warning(f"CSV Load Warning: {e}. Default lookups active.")

    try:
        sev_df = pd.read_csv(BASE_DIR / "severity_Africa20.csv")
        sev_dict = dict(zip(sev_df['Symptom'], sev_df['weight']))
    except Exception:
        sev_dict = {s: 1 for s in all_symptoms}

    return encoder, feature_columns, all_symptoms, diseases, disease_info, sev_dict

model = load_model()
encoder, feature_columns, all_symptoms, all_diseases, disease_info, symptom_severity = load_support_files()

# ========== HOME PAGE ==========
if st.session_state.page == "home":
    st.markdown("### Health Symptom Checker (HSC AI)")
    st.caption("Team - Healthcare Intelligence | Capstone Project")
    st.divider()

    col_welcome, col_stats = st.columns([3, 2], gap="large")

    with col_welcome:
        st.session_state.username = st.text_input(
            "Enter Your Name", 
            value=st.session_state.username if st.session_state.username != "Guest" else "", 
            placeholder="e.g. Abasifreke Udoh"
        )
        
        user_display = st.session_state.username.strip() if st.session_state.username.strip() else "Guest"
        st.session_state.username = user_display

        st.markdown(f"## Welcome, {user_display} 👋")
        st.markdown(
            "Get instant AI-driven medical predictions based on symptoms you select. "
            "For optimal accuracy, select between **4 to 6 symptoms**."
        )

        st.markdown("""
        <div class="how-it-works-box">
            <h4>💡 How it works:</h4>
            <ol>
                <li>Enter your name above to personalize your report</li>
                <li>Click <strong>'Start Diagnosis →'</strong> to begin</li>
                <li>Select <strong>4 to 6 symptoms</strong> from the search bar</li>
                <li>Get <strong>Top 3 candidate conditions</strong> with risk triage scores</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    with col_stats:
        st.write("")
        st.write("")
        s1, s2 = st.columns(2)
        with s1:
            st.markdown("""
            <div class="stat-card">
                <div class="stat-value">20</div>
                <div class="stat-label">Africa Diseases</div>
            </div>
            """, unsafe_allow_html=True)
        with s2:
            st.markdown("""
            <div class="stat-card">
                <div class="stat-value">80</div>
                <div class="stat-label">Tracked Symptoms</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        st.info("ℹ️ **Educational Note:** Designed for preliminary healthcare triage and educational demonstration.")

    st.divider()
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        st.button("Start Diagnosis →", type="primary", use_container_width=True, on_click=go_to_checker)

    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>© 2026 HEALTHCARE INTELLIGENCE • TechCrush Cohort 7<br>This application is for educational demonstration only. Always consult a qualified physician.</div>", unsafe_allow_html=True)
    st.stop()

# ========== CHECKER PAGE ==========
st.sidebar.button("← Back to Home", on_click=go_to_home, use_container_width=True)

st.title(f"🩺 Symptom Diagnosis for {st.session_state.username}")
st.markdown("Select your current symptoms to evaluate potential medical conditions.")

# ========== USER INPUT SECTION ==========
st.subheader("Step 1: Select Symptoms")
col_input, col_clear = st.columns([4, 1])

with col_input:
    display_names = {s: s.replace("_", " ").title() for s in all_symptoms}
    
    selected_symptoms = st.multiselect(
        "Search & select symptoms you are experiencing:",
        options=sorted(all_symptoms),
        format_func=lambda x: display_names.get(x, x),
        placeholder="Type to search symptoms (e.g. fever, headache)...",
        max_selections=6,
        key=f"symptom_selector_{st.session_state.clear_counter}"
    )

with col_clear:
    st.write("")
    st.write("")
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.clear_counter += 1
        st.session_state.prediction_made = False
        st.rerun()

if selected_symptoms:
    st.success(f"✓ Selected **{len(selected_symptoms)} / 6** symptoms: " + ", ".join([display_names[s] for s in selected_symptoms]))

# ========== PREDICTION CALCULATION ==========
if st.button("🔍 Predict Candidate Diseases", type="primary", use_container_width=True):
    if len(selected_symptoms) < 4:
        st.warning("⚠️ Please select at least **4 symptoms** for reliable prediction.")
    else:
        with st.spinner("Analyzing symptom vector..."):
            input_vector = [1 if s in selected_symptoms else 0 for s in all_symptoms]
            input_df = pd.DataFrame([input_vector], columns=all_symptoms)
            probs = model.predict_proba(input_df)[0]

            # Probability Calibration
            probs = probs ** 0.55
            probs = probs / probs.sum()

            top3_idx = np.argsort(probs)[-3:][::-1]

            st.session_state.prediction_made = True
            st.session_state.prediction = all_diseases[top3_idx[0]]
            st.session_state.top3_diseases = [all_diseases[i] for i in top3_idx]
            st.session_state.top3_probs = probs[top3_idx]
            st.session_state.selected_symptoms = selected_symptoms

# ========== RESULTS DISPLAY ==========
if st.session_state.prediction_made:
    st.divider()
    st.subheader("Step 2: Diagnostic Results & Triage")

    selected_symptoms = st.session_state.selected_symptoms
    num_symptoms = len(selected_symptoms)
    total_severity = sum([symptom_severity.get(s, 1) for s in selected_symptoms])

    # Severity Rating Rule Engine
    if num_symptoms >= 5 or total_severity >= 12:
        risk_text, risk_class = "High Risk", "severity-high"
    elif num_symptoms >= 3 or total_severity >= 6:
        risk_text, risk_class = "Moderate Risk", "severity-moderate"
    else:
        risk_text, risk_class = "Low Risk", "severity-low"

    conf = st.session_state.top3_probs[0] * 100

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f'''
        <div class="hero-prediction-card">
            <h3>Most Likely Condition</h3>
            <h2>{st.session_state.prediction}</h2>
        </div>
        ''', unsafe_allow_html=True)
    with col2:
        st.metric("Model Confidence", f"{conf:.0f}%")
    with col3:
        st.markdown(f'''
        <div class="severity-badge {risk_class}">
            <div style="font-size: 0.8rem; text-transform: uppercase;">Triage Severity</div>
            <div style="font-size: 1.4rem; font-weight: 700; margin-top: 4px;">{risk_text}</div>
        </div>
        ''', unsafe_allow_html=True)

    info = disease_info.get(st.session_state.prediction, {'description': 'No description found.', 'precautions': ["Consult a physician."]})
    
    with st.expander(f"📖 Medical Information & Actionable Precautions for {st.session_state.prediction}", expanded=True):
        st.markdown("**Description:**")
        st.write(info['description'])
        st.markdown("**Recommended Precautions:**")
        for i, p in enumerate(info['precautions'], 1):
            st.markdown(f"{i}. {p}")

    st.subheader("📊 Top 3 Candidate Diagnoses")
    for i in range(3):
        disease = st.session_state.top3_diseases[i]
        prob = st.session_state.top3_probs[i]
        with st.container(border=True):
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(f"**{i+1}. {disease}**")
                st.progress(prob)
            with col_b:
                st.metric(label="Probability", value=f"{prob*100:.1f}%")

    st.error("⚠️ **Medical Disclaimer:** This prediction is generated by an educational Machine Learning model. It is NOT a clinical diagnosis. Always consult a licensed medical professional for treatment.")

    # ========== FEEDBACK LOGGING ==========
    st.divider()
    st.subheader("📝 Feedback & Audit")
    feedback_file = "feedback.csv"
    if not os.path.exists(feedback_file):
        pd.DataFrame(columns=["timestamp", "username", "symptoms", "predicted", "num_symptoms", "total_severity", "feedback", "correct_disease"]).to_csv(feedback_file, index=False, sep='|')

    fb_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "username": st.session_state.username,
        "symptoms": ";".join(selected_symptoms),
        "predicted": st.session_state.prediction,
        "num_symptoms": num_symptoms,
        "total_severity": total_severity,
        "feedback": "",
        "correct_disease": ""
    }

    c1, c2 = st.columns(2)
    with c1:
        if st.button("👍 Helpful", use_container_width=True):
            fb_data["feedback"] = "Helpful"
            pd.DataFrame([fb_data]).to_csv(feedback_file, mode='a', header=False, index=False, sep='|', quoting=csv.QUOTE_ALL)
            st.success("Thank you for your feedback!")
    with c2:
        if st.button("👎 Not Helpful", use_container_width=True):
            fb_data["feedback"] = "Not Helpful"
            pd.DataFrame([fb_data]).to_csv(feedback_file, mode='a', header=False, index=False, sep='|', quoting=csv.QUOTE_ALL)
            st.warning("Feedback logged.")

    correct = st.text_input("If incorrect, enter the actual condition:")
    if st.button("Submit Correction", use_container_width=True):
        if correct.strip():
            fb_data["feedback"] = "Correction"
            fb_data["correct_disease"] = correct.strip()
            pd.DataFrame([fb_data]).to_csv(feedback_file, mode='a', header=False, index=False, sep='|', quoting=csv.QUOTE_ALL)
            st.success("Correction saved to audit log!")
        else:
            st.error("Please enter a disease name.")

# ========== SIDEBAR ADMIN PORTAL ==========
with st.sidebar:
    st.divider()
    with st.expander("🔒 Staff Portal"):
        admin_pass = st.text_input("Access Code", type="password", key="admin_pass")
        if st.button("Login", use_container_width=True):
            if admin_pass == "HCI2026":
                st.session_state.admin_logged_in = True
                st.success("Access Granted")
                st.rerun()
            else:
                st.error("Invalid Access Code")

if st.session_state.admin_logged_in:
    with st.sidebar:
        st.success("Logged In as Staff")
        st.subheader("Feedback Audit Dashboard")
        if os.path.exists("feedback.csv"):
            try:
                df_fb = pd.read_csv("feedback.csv", sep='|', on_bad_lines='skip')
                st.metric("Total User Responses", len(df_fb))
                st.dataframe(df_fb.tail(10), use_container_width=True)
                with open("feedback.csv", "rb") as f:
                    st.download_button("📥 Download Audit CSV", f, "feedback.csv")
            except Exception as e:
                st.error(f"CSV Read Error: {e}")
        else:
            st.write("No feedback logged yet.")
            
        if st.button("Logout", use_container_width=True):
            st.session_state.admin_logged_in = False
            st.rerun()

# ========== FOOTER ==========
st.markdown("---")
st.markdown("<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>© 2026 HEALTHCARE INTELLIGENCE • TechCrush Cohort 7<br>This application is for educational demonstration only. Always consult a qualified physician.</div>", unsafe_allow_html=True)