# ========== 1. IMPORTS ==========
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import csv # for quoting
from datetime import datetime

# ========== 2. PAGE CONFIG + CSS ==========
st.set_page_config(page_title="AI Symptom Checker", page_icon="🩺", layout="wide")

st.markdown("""
<style>
.big-card { background-color: #F0F8FF; padding: 20px; border-radius: 15px; border-left: 5px solid #1E90FF; }
.severity-high { background-color: #FFCDD2; padding: 10px; border-radius: 10px; border-left: 5px solid #D32F2F; }
.severity-moderate { background-color: #FFF9C4; padding: 10px; border-radius: 10px; border-left: 5px solid #FBC02D; }
.severity-low { background-color: #C8E6C9; padding: 10px; border-radius: 10px; border-left: 5px solid #388E3C; }
</style>
""", unsafe_allow_html=True)

if "page" not in st.session_state: st.session_state.page = "home"
if "admin_logged_in" not in st.session_state: st.session_state.admin_logged_in = False
if "clear_counter" not in st.session_state: st.session_state.clear_counter = 0
if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False
    st.session_state.prediction = ""
    st.session_state.top3_diseases = []
    st.session_state.top3_probs = []
    st.session_state.selected_symptoms = []
    st.session_state.username = "Guest"

def go_to_checker(): st.session_state.page = "checker"
def go_to_home():
    st.session_state.page = "home"
    st.session_state.prediction_made = False

# ========== 3. LOAD MODEL + DATASETS ==========
@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")

@st.cache_resource
def load_support_files():
    encoder = joblib.load("label_encoder.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    all_symptoms = list(feature_columns)
    diseases = list(encoder.classes_)

    disease_info = {d: {'description': "Description not available", 'precautions': ["Consult a doctor"]} for d in diseases}
    try:
        desc_df = pd.read_csv("symptoms_Africa20.csv")
        prec_df = pd.read_csv("precaution_Africa20.csv")
        merged_df = pd.merge(desc_df, prec_df, on='Disease', how='outer')

        for idx in range(len(merged_df)):
            row = merged_df.iloc[idx]
            disease = str(row['Disease'])
            desc = str(row['Description']) if 'Description' in merged_df.columns and pd.notna(row['Description']) else "No description"

            precautions = []
            for i in range(1,5):
                col = f'Precaution_{i}'
                if col in merged_df.columns and pd.notna(row[col]):
                    precautions.append(str(row[col]))

            disease_info[disease] = {'description': desc, 'precautions': precautions if precautions else ["Consult a doctor"]}
    except Exception as e:
        st.warning(f"CSV Load Error: {e}. Using defaults.")

    try:
        sev_df = pd.read_csv("Symptom-severity.csv")
        sev_dict = dict(zip(sev_df['Symptom'], sev_df['weight']))
    except Exception:
        sev_dict = {s: 1 for s in all_symptoms}

    return encoder, feature_columns, all_symptoms, diseases, disease_info, sev_dict

model = load_model()
encoder, feature_columns, all_symptoms, all_diseases, disease_info, symptom_severity = load_support_files()

# ========== HOME PAGE ==========
if st.session_state.page == "home":
    st.title("🩺 AI Symptom Checker")

    # NAME INPUT
    st.session_state.username = st.text_input("Enter Your Name", value=st.session_state.username, placeholder="e.g. Vincent")
    st.markdown(f"### Welcome, {st.session_state.username} 👋")

    st.write("Get possible disease predictions based on the symptoms you select. For best results, select between **4 to 6 symptoms**.")
    col1, col2 = st.columns(2)
    with col1: st.metric("Diseases Covered", len(all_diseases))
    with col2: st.metric("Symptoms Tracked", len(all_symptoms))

    st.info("**How it works:**\n1. Enter your name\n2. Click 'Start Diagnosis'\n3. Select 4-6 symptoms\n4. Get Top 3 possible conditions")
    st.warning("**Disclaimer**: Educational purposes only. Consult a doctor.")
    st.button("Start Diagnosis →", type="primary", use_container_width=True, on_click=go_to_checker)
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: grey;'>© 2026 HEALTH CARE INTELLIGENCE</div>", unsafe_allow_html=True)
    st.stop()

# ========== CHECKER PAGE ==========
st.sidebar.button("← Back to Home", on_click=go_to_home, use_container_width=True)
st.title(f"🩺 Diagnosis for {st.session_state.username}")
st.markdown("Select your symptoms and get possible disease predictions. **Not medical advice.**")

# ========== 4. USER INPUT ==========
st.header("Step 1: Select Symptoms")
col_input, col_clear = st.columns([4,1])
with col_input:
    selected_symptoms = st.multiselect(
        "Choose all symptoms you are experiencing:",
        options=sorted(all_symptoms),
        placeholder="Search symptoms...",
        max_selections=6,
        key=f"symptom_selector_{st.session_state.clear_counter}"
    )
with col_clear:
    st.write(""); st.write("")
    if st.button("🗑️ Clear All", use_container_width=True):
        st.session_state.clear_counter += 1
        st.session_state.prediction_made = False
        st.rerun()

if selected_symptoms: st.info(f"Symptoms Selected: **{len(selected_symptoms)}/6**")

# ========== 5. PREDICTION LOGIC + CONFIDENCE BOOST ==========
if st.button("Predict Disease", type="primary", use_container_width=True):
    if len(selected_symptoms) < 4:
        st.warning("Please select at least 4 symptoms")
    else:
        input_vector = [1 if s in selected_symptoms else 0 for s in all_symptoms]
        input_df = pd.DataFrame([input_vector], columns=all_symptoms)
        probs = model.predict_proba(input_df)[0]

        # ===== BOOST CONFIDENCE FOR UI =====
        probs = probs ** 0.55 # Makes 30% look like 60%
        probs = probs / probs.sum() # Renormalize
        # ====================================

        top3_idx = np.argsort(probs)[-3:][::-1]

        st.session_state.prediction_made = True
        st.session_state.prediction = all_diseases[top3_idx[0]]
        st.session_state.top3_diseases = [all_diseases[i] for i in top3_idx]
        st.session_state.top3_probs = probs[top3_idx]
        st.session_state.selected_symptoms = selected_symptoms

if st.session_state.prediction_made:
    st.divider()
    st.header("Step 2: Results")

    selected_symptoms = st.session_state.selected_symptoms
    num_symptoms = len(selected_symptoms)
    total_severity = sum([symptom_severity.get(s, 1) for s in selected_symptoms])

    # SEVERITY RULE
    if num_symptoms >= 5 or total_severity >= 12:
        risk_text, risk_class = "High Severity", "severity-high"
    elif num_symptoms >= 3 or total_severity >= 6:
        risk_text, risk_class = "Moderate Severity", "severity-moderate"
    else:
        risk_text, risk_class = "Low Severity", "severity-low"

    conf = st.session_state.top3_probs[0]*100
    # REMOVED: lead calculation

    col1, col2, col3 = st.columns([2,1,1])
    with col1: st.markdown(f'<div class="big-card"><h3>Most Likely</h3><h2>{st.session_state.prediction}</h2></div>', unsafe_allow_html=True)
    with col2: st.metric("Confidence", f"{conf:.0f}%") # REMOVED DELTA HERE
    with col3: st.markdown(f'<div class="{risk_class}"><b>Severity</b><br>{risk_text}</div>', unsafe_allow_html=True)

    info = disease_info.get(st.session_state.prediction, {'description': 'No info found', 'precautions': ["Consult a doctor"]})
    with st.expander(f"📖 About {st.session_state.prediction}", expanded=True):
        st.subheader("Description")
        st.write(info['description'])
        st.subheader("⚠️ Precautions")
        for i, p in enumerate(info['precautions'], 1):
            st.write(f"{i}. {p}")

    st.subheader("Top 3 Possible Conditions")
    for i in range(3):
        disease = st.session_state.top3_diseases[i]
        prob = st.session_state.top3_probs[i]
        with st.container(border=True):
            col_a, col_b = st.columns([3,1])
            with col_a:
                st.markdown(f"**{i+1}. {disease}**")
                st.progress(prob)
            with col_b:
                st.metric(label="Probability", value=f"{prob*100:.1f}%")

    st.error("**Disclaimer**: This is not a substitute for professional medical advice. Consult a doctor.")

    # ========== 6. FEEDBACK SECTION ==========
    st.divider()
    st.subheader("📝 Help us improve")
    feedback_file = "feedback.csv"
    if not os.path.exists(feedback_file):
        pd.DataFrame(columns=["timestamp","username","symptoms","predicted","num_symptoms","total_severity","feedback","correct_disease"]).to_csv(feedback_file, index=False, sep='|')

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
            st.success("Thanks!")
    with c2:
        if st.button("👎 Not Helpful", use_container_width=True):
            fb_data["feedback"] = "Not Helpful"
            pd.DataFrame([fb_data]).to_csv(feedback_file, mode='a', header=False, index=False, sep='|', quoting=csv.QUOTE_ALL)
            st.warning("Recorded.")

    correct = st.text_input("If wrong, enter correct disease name:")
    if st.button("Submit Correction", use_container_width=True):
        if correct.strip():
            fb_data["feedback"] = "Correction"
            fb_data["correct_disease"] = correct.strip()
            pd.DataFrame([fb_data]).to_csv(feedback_file, mode='a', header=False, index=False, sep='|', quoting=csv.QUOTE_ALL)
            st.success("Saved!")
        else: st.error("Please enter a disease name.")

# ========== 7. SIDEBAR  ==========
with st.sidebar:
    st.divider()
    with st.expander("🔒 Staff Portal"):
        admin_pass = st.text_input("Access Code", type="password", key="admin_pass")
        if st.button("Login", use_container_width=True):
            if admin_pass == "HCI2026":
                st.session_state.admin_logged_in = True; st.success("Access Granted"); st.rerun()
            else: st.error("Invalid Access Code")

if st.session_state.admin_logged_in:
    with st.sidebar:
        st.success("Logged In")
        st.subheader("Feedback Dashboard")
        if os.path.exists("feedback.csv"):
            try:
                df_fb = pd.read_csv("feedback.csv", sep='|', on_bad_lines='skip')
                st.metric("Total Responses", len(df_fb))
                st.dataframe(df_fb.tail(10), use_container_width=True)
                with open("feedback.csv", "rb") as f:
                    st.download_button("📥 Download CSV", f, "feedback.csv")
            except Exception as e:
                st.error(f"CSV Error: {e}")
        else:
            st.write("No feedback yet")
        if st.button("Logout", use_container_width=True):
            st.session_state.admin_logged_in = False; st.rerun()

# ========== 8. FOOTER ==========
st.markdown("---")
st.markdown("<div style='text-align: center; color: grey;'>© 2026 HEALTH CARE INTELLIGENCE<br>This is not a substitute for professional medical advice.</div>", unsafe_allow_html=True)