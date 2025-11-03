# 🔍 Anomaly Detection

This notebook uses **PyCaret’s anomaly detection module** to find unusual or abnormal data points in a dataset.

## 📘 Overview
- Used **Isolation Forest** (`create_model('iforest')`)
- Labeled anomalies using `assign_model()`
- Visualized data clusters using a **t-SNE plot**
- Saved the model and labeled results for further analysis

## ⚙️ Steps
1. Loaded and cleaned the dataset (numeric columns only)
2. Initialized PyCaret using `setup()`
3. Created and trained the anomaly detection model
4. Visualized clusters and saved labeled data

## 📊 Outcome
- Identified outliers and visualized their separation
- Exported `best_anomaly_model.pkl` and `anomaly_labeled.csv`
