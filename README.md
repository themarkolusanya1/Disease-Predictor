# 🌍 Disease Predictor (AI Symptom Checker)

A machine learning project designed to predict likely medical conditions from user-reported symptoms. Developed by Team - Healthcare Intelligence, the project focuses on **20 key diseases highly relevant to Africa**, providing quick health insights, top 3 differential diagnoses, disease descriptions, risk severity ratings, and recommended precautions via an interactive Streamlit web application.

---

## 📌 Key Application Features

* 🩺 **User Onboarding**: Personal diagnosis workflow with user name personalization.
* 🔍 **Multiselect Symptom Search**: Search and select up to 6 symptoms from 80 tracked health indicators.
* 📊 **Top 3 Differential Diagnoses**: Displays the top 3 candidate diseases with probability scores and calibrated confidence progress bars.
* ⚠️ **Risk Severity Triage**: Evaluates patient symptom severity (*Low*, *Moderate*, *High Risk*) using symptom count and weighted severity scores.
* 📖 **Medical Info & Precautions**: Provides instant disease descriptions and step-by-step next action precautions.
* 📝 **User Feedback System**: Users can rate predictions (👍 Helpful / 👎 Not Helpful) or submit corrections, logged to `feedback.csv`.
* 🔒 **Staff/Admin Portal**: Passcode-protected portal (`HCI2026`) for healthcare administrators to inspect feedback metrics and export CSV data.

---

## 🦠 Target Diseases (20 Africa-Relevant Focus)

The model is trained to detect and classify the following 20 diseases:

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

## 📂 Project Structure

```text
Disease-Predictor/
├── app.py                      # Main Streamlit web application (v2 AI Symptom Checker)
├── best_model.pkl              # Deployed Random Forest Classifier model
├── feature_columns.pkl         # Serialized 80 binary symptom features
├── label_encoder.pkl           # Categorical label encoder for 20 target diseases
├── model_comparison.csv        # Evaluated ML model performance metrics
├── requirements.txt       # Python project dependencies
├── runtime.txt                 # Python runtime specification (python-3.11)
├── README.md                   # Comprehensive project documentation
├── symptoms_Africa20.csv       # Disease descriptions lookup table
├── precaution_Africa20.csv     # Recommended medical precautions lookup table
├── severity_Africa20.csv       # Symptom severity weight definitions
├── Task5_Model_Training.ipynb # ML pipeline, training notebooks & evaluation
├── notebook.ipynb              # Exploratory Data Analysis (EDA) notebook
└── data/
    ├── raw/                    # Original raw source datasets
    └── processed/              # Filtered dataset (dataset_Africa20.csv - 146 samples)
```

---

## 🧪 Machine Learning Pipeline & Evaluation

### 1. Dataset & Preprocessing
* **Filtered Dataset**: Extracted 146 samples across the 20 target diseases with 80 unique binary symptom features (`data/processed/dataset_Africa20.csv`).
* **Class Balancing**: Balanced target classes (~5–10 samples per disease) using `class_weight="balanced"` in scikit-learn classifiers.

### 2. Model Performance Comparison

Multiple classification algorithms were trained and evaluated on an 80/20 train-test split:

| Model | Accuracy | Precision | Recall | F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Random Forest** | **1.00 (100%)** | **1.00** | **1.00** | **1.00** | 🏆 **Selected for Deployment** (`best_model.pkl`) |
| **SVM (Linear)** | **1.00 (100%)** | **1.00** | **1.00** | **1.00** | ⚡ Top Performer |
| **Logistic Regression** | **1.00 (100%)** | **1.00** | **1.00** | **1.00** | ⚡ Top Performer |
| **KNN** | **1.00 (100%)** | **1.00** | **1.00** | **1.00** | ⚡ Top Performer |
| **Decision Tree** | **0.57 (56.7%)** | **0.64** | **0.57** | **0.57** | ⚠️ Underperformed |

### 3. Model Selection & Overfitting Mitigation
* **Why Random Forest?**: Chosen as the primary model due to its ensemble tree stability, multi-class probability outputs (`predict_proba`), and robustness against sparse input vectors.
* **Mitigation Strategies**: To handle dataset limitations in production, the application employs **Top 3 Differential Diagnoses**, **Probability Smoothing** (`probs ** 0.55`), and **Risk Severity Triage**.

---

## 📋 Project Status & Team Milestones

- [x] **Data Filtering:** Extracted and filtered datasets down to 20 target African diseases.
- [x] **Data Cleaning & EDA:** Standardized symptom names, missing values, and frequency distributions.
- [x] **Feature Engineering & Preprocessing:** Encoded binary feature vectors for 80 symptoms and target disease labels.
- [x] **Model Training & Comparison:** Trained and benchmarked 5 machine learning models (Random Forest, SVM, KNN, Logistic Regression, Decision Tree).
- [x] **Model Export:** Serialized `best_model.pkl`, `feature_columns.pkl`, and `label_encoder.pkl`.
- [x] **Streamlit Web Application Integration:** Built and deployed the interactive v2 AI Symptom Checker UI (`app.py`).
- [x] **Documentation & Deployment:** Updated complete project documentation and deployed to Streamlit Community Cloud.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/themarkolusanya1/Disease-Predictor.git
cd Disease-Predictor
```

### 2. Install Dependencies
Ensure Python 3.8+ is installed, then run:
```bash
pip install -r requirements.txt
```

### 3. Running Locally
Launch the Streamlit web application:
```bash
streamlit run app.py
```

### 4. Running Jupyter Notebooks
To explore the model training pipeline:
```bash
jupyter notebook Task5_Model_Training.ipynb
```

---

## 👥 Credits & Acknowledgments

* **Development Team**: CAPSTONE Group 1
* **Department**: Department of Healthcare Intelligence
* **Year**: 2026
