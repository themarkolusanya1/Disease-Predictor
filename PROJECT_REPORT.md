# 🏥 CAPSTONE PROJECT FINAL REPORT
## AI-Powered Disease Predictor for Healthcare Intelligence

**Department:** Department of Healthcare Intelligence  
**Team:** CAPSTONE Group 1  
**Project Title:** AI Disease Predictor & Symptom Checker  
**Target Focus:** 20 Key Africa-Relevant Diseases  
**Deployment Platform:** Streamlit Community Cloud  
**Academic Year:** 2026  

---

## 📋 EXECUTIVE SUMMARY

In many healthcare settings across Africa, access to preliminary diagnostic triage and immediate medical guidance remains limited. The **AI Disease Predictor** project addresses this challenge by engineering an interactive machine learning classification platform designed to evaluate user-reported symptoms and provide instant, preliminary health insights.

Focused specifically on 20 diseases highly prevalent in regional healthcare contexts (including Malaria, Typhoid, Tuberculosis, HIV/AIDS, and Pneumonia), the system integrates an ensemble **Random Forest Classifier** with a modern **Streamlit** web application. The platform provides differential diagnoses (Top 3 candidates), risk severity triage (Low, Moderate, High Risk), medical descriptions, actionable precautions, and a closed-loop user feedback portal for continuous model evaluation.

---

## 1. PROJECT OBJECTIVES & SCOPE

1. **Targeted Disease Matrix:** Filter and specialize machine learning training around 20 critical diseases relevant to African healthcare.
2. **Feature Standardization:** Map complex symptom patterns into an 80-feature binary vector space ($1$ = symptom present, $0$ = absent).
3. **Model Benchmarking:** Train, evaluate, and compare multiple machine learning algorithms (Random Forest, SVM, KNN, Logistic Regression, Decision Tree) on key evaluation metrics (Accuracy, Precision, Recall, F1-Score).
4. **Interactive Web Application:** Build an end-to-end web application with user onboarding, multiselect symptom search, risk severity scoring, and feedback logging.
5. **Production Deployment:** Host the production model (`best_model.pkl`) on a cloud platform (Streamlit Community Cloud) for 24/7 public accessibility.

---

## 2. SYSTEM ARCHITECTURE & PIPELINE

```text
┌─────────────────────────┐
│ User Symptom Selection  │  (Select 1 to 6 symptoms out of 80 features)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Binary Vectorization  │  (Converts selected symptoms to [0, 1, 0, ...] vector)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Random Forest Model     │  (best_model.pkl trained with class_weight="balanced")
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Probability Calibration │  (probs ** 0.55 probability smoothing)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Differential Diagnosis  │  (Top 3 Disease Candidates + Confidence Progress Bars)
│ Risk Triage Score       │  (Low / Moderate / High Severity Rating)
│ Feedback Logging        │  (Stores responses in feedback.csv for admin audit)
└─────────────────────────┘
```

---

## 3. TARGET DISEASE MATRIX (20 DISEASES)

The machine learning classifier is specialized to identify the following 20 conditions:

1. **Malaria**
2. **Typhoid**
3. **Tuberculosis (TB)**
4. **AIDS (HIV)**
5. **Gastroenteritis**
6. **Jaundice**
7. **Pneumonia**
8. **Common Cold**
9. **Urinary Tract Infection (UTI)**
10. **Diabetes**
11. **Hypertension**
12. **Hypoglycemia**
13. **Chicken Pox**
14. **Drug Reaction**
15. **Allergy**
16. **Bronchial Asthma**
17. **Arthritis**
18. **Migraine**
19. **Heart Attack**
20. **Paralysis (Brain Hemorrhage)**

---

## 4. MACHINE LEARNING METHODOLOGY & RESULTS

### 4.1 Dataset & Feature Engineering
* **Processed Dataset Size:** 146 samples across 20 target classes (`data/processed/dataset_Africa20.csv`).
* **Feature Dimensionality:** 80 unique binary symptom features derived from symptom frequency analysis.
* **Class Balance Strategy:** Configured `class_weight="balanced"` to handle minor class sample variations (5 to 10 samples per disease).

### 4.2 Model Performance Benchmarking

Five classification algorithms were trained on an 80/20 train-test split and evaluated on standard metrics:

| Classification Model | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | Selection Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Random Forest Classifier** | **1.00 (100%)** | **1.00** | **1.00** | **1.00** | 🏆 **Deployed Model** (`best_model.pkl`) |
| **Support Vector Machine (Linear)** | **1.00 (100%)** | **1.00** | **1.00** | **1.00** | ⚡ Top Performer |
| **Logistic Regression** | **1.00 (100%)** | **1.00** | **1.00** | **1.00** | ⚡ Top Performer |
| **K-Nearest Neighbors (KNN)** | **1.00 (100%)** | **1.00** | **1.00** | **1.00** | ⚡ Top Performer |
| **Decision Tree Classifier** | **0.57 (56.7%)** | **0.64** | **0.57** | **0.57** | ⚠️ Baseline Benchmark |

### 4.3 Rationale for Model Selection
**Random Forest** (`n_estimators=100`, `class_weight="balanced"`) was chosen for production deployment over linear SVM and Logistic Regression because:
1. **Multi-Class Probability Estimation:** Random Forest natively computes smooth probability vectors via `predict_proba()`, enabling differential top-3 diagnosis bars.
2. **Variance Reduction:** Combining 100 decision trees mitigates variance issues inherent in single decision trees.
3. **Handling Sparse Inputs:** Demonstrates superior stability when users select incomplete symptom lists (e.g., 2–3 symptoms out of 80).

---

## 5. APPLICATION DESIGN & UX FEATURES

The application (`app.py`) includes several key features designed for medical utility and user safety:

1. **Searchable Multiselect UI:** Users can search and select up to 6 symptoms directly via an interactive search bar.
2. **Top 3 Differential Diagnoses:** Instead of a single rigid prediction, the app presents the top 3 most probable conditions with calibrated probability bars.
3. **Risk Severity Triage Engine:** Calculates patient risk levels (*Low*, *Moderate*, *High Risk*) using symptom count thresholding and weighted severity lookup tables (`severity_Africa20.csv`).
4. **Closed-Loop Feedback Portal:** Captures user validation (👍 Helpful / 👎 Not Helpful / Correction submission) appended to `feedback.csv`.
5. **Passcode-Protected Admin Portal:** Accessible via access code `HCI2026`, allowing healthcare administrators to audit real-time user feedback metrics.

---

## 6. OVERFITTING & PRODUCTION CONSIDERATIONS

Due to the initial benchmark dataset size (146 samples), the model achieved 100% test accuracy on synthetic pattern combinations. To ensure safety and reliability in live environments, the system incorporates:

* **Probability Calibration (`probs ** 0.55`):** Smoothes hyper-confident binary predictions into realistic differential distributions.
* **Top 3 Candidate Display:** Reduces risk of single-point diagnostic failure.
* **Strict Medical Disclaimers:** Explicitly highlights that the app is an educational prototype and not a formal medical diagnostic tool.

---

## 7. RECOMMENDATIONS FOR FUTURE WORK

1. **Dataset Expansion:** Retrain the model on large-scale clinical electronic health records (EHR) containing 5,000+ patient encounters to improve generalizability.
2. **Symptom Weighting & Duration:** Incorporate symptom duration (e.g., *fever lasting > 7 days*) and severity scales (1–10).
3. **Multilingual Support:** Translate UI into regional languages (e.g., Swahili, Hausa, Yoruba, French) for broader accessibility across healthcare centers.

---

## 8. CONCLUSION

The **AI Disease Predictor** project successfully demonstrates the feasibility of combining machine learning classifiers with an intuitive web application to deliver rapid, preliminary health triage for 20 Africa-relevant diseases. The deployed platform offers a solid foundation for future AI-driven telehealth tools.

---
**Report Compiled By:** CAPSTONE Group 1  
**Department:** Department of Healthcare Intelligence  
**Repository:** [github.com/themarkolusanya1/Disease-Predictor](https://github.com/themarkolusanya1/Disease-Predictor)  
