# Disease Predictor


A machine learning project that predicts likely diseases from a set of user-reported symptoms, focused on 20 diseases relevant to Africa. Includes a Streamlit web app where users enter symptoms and receive a prediction along with a short description and precautions.

## Project Structure
```
Disease-Predictor/
├── notebook.ipynb       # Data loading, cleaning, EDA, training
├── app.py                # Streamlit web app
├── data/
│   ├── raw/               # Original source datasets
│   └── processed/         # Filtered 20-disease datasets
└── README.md
```

## Team Tasks
- Import, load, and filter data to 20 diseases
- Data cleaning
- Exploratory data analysis (EDA)
- Data preprocessing
- Model training and evaluation
- Streamlit app (model integration, prediction, UI)
- Research and Q&A prep

## Getting Started
1. Clone the repo: `git clone https://github.com/themarkolusanya1/Disease-Predictor.git`
2. Install dependencies: `pip install pandas numpy scikit-learn streamlit`
3. Run the notebook to explore the data pipeline, or run the app: `streamlit run app.py`

## Contributing
- Create your own branch before making changes: `git checkout -b your-name-task`
- Push your branch and open a pull request into `main`
- Wait for review and approval before merging