# 🏡 AutoGluon — California House Prices
### Part 1(a) of AutoGluon Assignment  
**Submitted by:** Dev Mulchandani  

---

## 📘 Overview
In this project, I built a **regression model** to predict house prices using AutoGluon’s `TabularPredictor`.  
The dataset used was the **California Housing Prices** dataset from Kaggle.  
AutoGluon automatically handled data preprocessing, feature selection, and model training, allowing me to build a strong model with minimal code.

---

## ⚙️ Steps Performed
1. Installed and imported AutoGluon and Kaggle.
2. Connected to Kaggle using the API key (`kaggle.json`).
3. Downloaded and extracted the California House Prices dataset.
4. Loaded data into Pandas and defined the **target column (`Sold Price`)**.
5. Trained a regression model using:
   - Preset: `medium_quality`
   - Time limit: 15 minutes
6. Evaluated model performance and generated predictions.
7. Exported results as `my_submission.csv` for Kaggle submission.

---

## 🧠 Key Features
- Automatic preprocessing (missing value handling, encoding, etc.)
- Model ensemble creation for improved accuracy
- Minimal manual coding
- Fast and efficient AutoML workflow

---

## 🧾 Files in this Folder
- `AutoGluon_House_Prices.ipynb` – Google Colab notebook  
- `Report.pdf` – Report with explanation and screenshots  

---

## 🏁 Results
AutoGluon produced strong results with minimal setup and automatically created a leaderboard of models ranked by performance. The model predictions were successfully submitted to Kaggle for evaluation.
