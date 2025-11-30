# CMPE255_Final_Project – Auto Data Toolkit (Titanic + CSV)

This repository contains the final project for **CMPE 255**, implemented as a
Streamlit web app and a small reusable Python module.

The app follows the **CRISP-DM** workflow on the classic Titanic dataset (and
optionally any user-uploaded CSV):

1. **Business Understanding** – Predict an outcome (e.g., survival on Titanic)
2. **Data Understanding** – Data preview, basic statistics, missing values
3. **Data Preparation** – Numeric/categorical splits, scaling, one-hot encoding
4. **Modeling** – RandomForest classifier in a scikit-learn `Pipeline`
5. **Evaluation** – Accuracy, precision, recall, F1, confusion matrix, ROC AUC
6. **Explainability** – SHAP feature importance
7. **Insights** – Partial Dependence Plots (PDP) for key numeric features

---

## 1. Project Structure

```text
CMPE255_Final_Project/
├── app.py               # Main Streamlit app (Titanic + CSV upload)
├── requirements.txt     # Python dependencies (for Streamlit Cloud / pip)
├── .gitignore
├── README.md
└── project_pipeline/
    ├── __init__.py
    └── core.py         # Small reusable experiment pipeline
```

---

## 2. How to run locally

### Create and activate a virtual environment (optional but recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # on macOS / Linux
# or on Windows:
# .venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit app

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

---

## 3. How to deploy on Streamlit Cloud

1. Push this folder as a GitHub repository, e.g. `CMPE255_Final_Project`.
2. Go to [https://share.streamlit.io](https://share.streamlit.io)
3. Click **New app**.
4. Select your repo, branch, and set:
   - **Main file path**: `app.py`
5. Click **Deploy**.

Streamlit Cloud will read `requirements.txt`, install dependencies, and launch
the app. The public URL can be submitted as the project demo link.

---

## 4. Using the app

When you open the app, you can:

1. Choose **Titanic demo** or **Upload CSV** from the sidebar.
2. Inspect the raw data preview and missing value summary.
3. Select the **target column**.
4. Adjust the **train/test split**.
5. Click **Run Auto-Toolkit**:
   - A RandomForest model is trained inside a scikit-learn `Pipeline`.
   - Metrics and visualizations are displayed:
     - Accuracy, precision (weighted), recall (weighted), F1 (weighted)
     - Confusion matrix
     - ROC curve (for binary problems)
     - Global feature importance (RandomForest)
     - SHAP feature importance (global)
     - Partial Dependence Plots (PDP) for key numeric features

---

## 5. Python API (project_pipeline)

For completeness and to satisfy the “Python package / module” requirement, the
`project_pipeline` package exposes a tiny API:

```python
from project_pipeline import run_full_experiment

result = run_full_experiment(df, target_col="Survived")
print(result.metrics)
model = result.model
```

This can be used from Jupyter notebooks or other scripts to reproduce the main
experiment outside of Streamlit.

---

## 6. Notes

- SHAP and PDP are computed on a sample of the training data to keep the app
  responsive.
- The Titanic demo dataset is loaded from a public GitHub URL and lightly
  cleaned by dropping very high-cardinality text columns (`Name`, `Ticket`,
  `Cabin`).
- When uploading your own CSV, you should:
  - Ensure the target column is present and selected.
  - Avoid extremely wide free-text columns for best model performance.
