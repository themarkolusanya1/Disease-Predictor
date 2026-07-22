# 🌍 Disease Predictor

A machine learning project designed to predict likely diseases from a set of user-reported symptoms. The project is focused on **20 key diseases highly relevant to Africa**, aiming to provide quick health insights, short disease descriptions, and recommended precautions via a modern Streamlit web application.

---

## 📂 Project Structure

```text
Disease-Predictor/
├── data/
│   ├── raw/               # Original source datasets
│   └── processed/         # Filtered 20-disease datasets
├── notebook.ipynb         # Data loading, cleaning, EDA, and model training
├── app.py                 # Streamlit web application UI & integration
├── requirements.txt       # Project python dependencies
└── README.md              # Project documentation
```

---

## 🦠 Target Diseases (Africa-Relevant Focus)

The model is tailored to detect and predict the following 20 diseases:
1. **Malaria**
2. **Typhoid**
3. **Tuberculosis (TB)**
4. **AIDS (HIV)**
5. **Gastroenteritis**
6. **Jaundice**
7. **Dengue**
8. **Pneumonia**
9. **Hepatitis A**
10. **Hepatitis B**
11. **Hepatitis C**
12. **Hepatitis D**
13. **Hepatitis E**
14. **Common Cold**
15. **Urinary Tract Infection (UTI)**
16. **Diabetes**
17. **Hypertension**
18. **Chickenpox**
19. **Drug Reaction**
20. **Allergy**

---

## 📋 Team Tasks & Progress

- [x] **Data Filtering:** Import, load, and filter datasets down to the 20 target diseases.
- [x] **Data Cleaning:** Handle missing values, duplicate entries, and symptom name formatting.
- [ ] **Exploratory Data Analysis (EDA):** Analyze symptom frequency, correlation matrices, and distribution.
- [ ] **Data Preprocessing:** Encode categorical symptom inputs and split data into train/test sets.
- [ ] **Model Training & Evaluation:** Train classifier models (e.g., Random Forest, Naive Bayes) and evaluate performance (Accuracy, F1-Score).
- [ ] **Streamlit App Integration:** Integrate the trained model, implement the prediction interface, and display descriptions & precautions.
- [ ] **Research & Q&A Prep:** Document findings and prepare for presentation.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/themarkolusanya1/Disease-Predictor.git
cd Disease-Predictor
```

### 2. Install Dependencies
Ensure you have Python 3.8+ installed, then run:
```bash
pip install -r requirements.txt
```
*(Dependencies: `pandas`, `numpy`, `scikit-learn`, `streamlit`)*

### 3. Running the Project
* **Explore the Machine Learning Pipeline:**
  Open and run the Jupyter notebook:
  ```bash
  jupyter notebook notebook.ipynb
  ```
* **Start the Streamlit Web Application:**
  Launch the web server locally:
  ```bash
  streamlit run app.py
  ```

---

## 🤝 Contributing

1. Create a branch for your task:
   ```bash
   git checkout -b your-name-task
   ```
2. Commit your changes with clear messages.
3. Push to your branch and open a **Pull Request** into `main`.
4. Wait for review and team approval before merging.