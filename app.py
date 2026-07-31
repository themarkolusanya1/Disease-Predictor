import streamlit as st
import joblib
import pandas as pd
st.set_page_config(page_title="Disease Predictor")
st.markdown("""
<style>
[data-testid="stSidebar"] {
    display: none;
}
</style>
""", unsafe_allow_html=True)
precautions = pd.read_csv("precaution_Africa20.csv")
descriptions = pd.read_csv("symptoms_Africa20.csv")

st.title("🩺 Disease Predictor")
st.caption("AI-powered disease prediction using logistic regression")
st.info(
    "This app predicts the disease using machine learning. it is for educational purposes only and should not replace medical advice."
)

model = joblib.load("best_model.pkl")
print(type(model))
encoder = joblib.load("label_encoder.pkl")
feature_columns = joblib.load("feature_columns.pkl")
if "prediction" not in st.session_state:
    st.session_state.prediction = None
    if "show_results" not in st.session_state:
         st.session_state.show_results = False

if not st.session_state.show_results:
    st.subheader("Select up to 6 symptoms")

symptom_options = [""] + sorted(feature_columns)
display_names = {
    symptom: symptom.replace("_", " ").title()
    for symptom in feature_columns
}

options = [""] + sorted(feature_columns)
symptom1 = st.selectbox("Symptom 1", options, format_func=lambda x: display_names.get(x, ""))
symptom2 = st.selectbox("Symptom 2", options, format_func=lambda x: display_names.get(x, ""))
symptom3 = st.selectbox("Symptom 3", options, format_func=lambda x: display_names.get(x, ""))
symptom4 = st.selectbox("Symptom 4", options, format_func=lambda x: display_names.get(x, ""))
symptom5 = st.selectbox("Symptom 5", options, format_func=lambda x: display_names.get(x, ""))
symptom6 = st.selectbox("Symptom 6", options, format_func=lambda x: display_names.get(x, ""))

col1, col2, col3 = st.columns([1,2,1])

with col2:
    predict = st.button("🔍 Predict Disease", use_container_width=True)

if predict:
    with st.spinner("🔍 Predicting disease..."):

        selected = [
            symptom1,
            symptom2,
            symptom3,
            symptom4,
            symptom5,
            symptom6,
        ]
        selected = [s for s in selected if s != ""]

        if len(selected) == 0:
            st.warning("⚠ Please select at least one symptom before predicting.")
            st.stop()

        input_vector = [0] * len(feature_columns)

        for symptom in selected:
         if symptom != "":
            index = feature_columns.index(symptom)
            input_vector[index] = 1

    input_df = pd.DataFrame([input_vector], columns=feature_columns)

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    confidence = max(probabilities) * 100

    disease = encoder.inverse_transform([prediction])[0]

    desc = descriptions[
    descriptions["Disease"].str.lower() == disease.lower()
]

    prec = precautions[
    precautions["Disease"].str.lower() == disease.lower()
]
    st.session_state.show_results = True
    st.session_state.prediction = {
        "disease": disease,
        "confidence": confidence,
        "desc" : desc,
        "prec" : prec
    }
    st.switch_page("pages/1_Results.py")
# ===========================
# SHOW RESULT PAGE
# ===========================