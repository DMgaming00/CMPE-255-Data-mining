# IPO Listing-Day Pop Prediction (CRISP-DM)

Predict whether an IPO lists with a positive first-day return using a leakage-safe, time-aware pipeline.

## Dataset
- Excel file (e.g., data/Initial Public Offering.xlsx)
- Typical fields: date, offer_price, list_price, listing_gain, qib, hni, rii, total, issue_size (crores)
- Kaggle link (optional): add here

## Methodology (CRISP-DM)
1. Data Understanding: schema audit, type parsing, distributions, relationships, time trends
2. Data Preparation: median imputation, IQR winsorization, IsolationForest flags, leakage guard
3. Feature Engineering: year/quarter/month/post_2020, demand shares & totals, log(size), interactions, light OHE
4. Modeling: chronological split; Logistic Regression, Random Forest, XGBoost (or GBM fallback), small MLP
5. Evaluation: Accuracy, Precision, Recall, F1, ROC-AUC; confusion matrices; ROC curves; generalization gap
6. Deployment Notes: save best model & feature schema to artifacts/

## How to Run
    python -m venv .venv
    # Windows: .venv\Scripts\activate
    # macOS/Linux:
    source .venv/bin/activate
    pip install -r requirements.txt
    python src/analysis.py --data "data/Initial Public Offering.xlsx"

## Structure
    ipo-listing-pop-crispdm/
    ├── data/
    ├── notebooks/
    ├── src/
    │   └── analysis.py
    ├── artifacts/
    ├── reports/
    ├── README.md
    └── requirements.txt

## Future Work
- Time-series CV (blocked/rolling), threshold tuning & calibration
- Macro/sector features; minimal FastAPI scoring + SHAP
