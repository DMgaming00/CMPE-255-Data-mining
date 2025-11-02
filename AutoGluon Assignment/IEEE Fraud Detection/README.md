# 💳 AutoGluon — IEEE Fraud Detection
### Part 1(b) of AutoGluon Assignment  
**Submitted by:** Dev Mulchandani  

---

## 📘 Overview
In this project, I built a machine-learning model to predict **fraudulent financial transactions** using AutoGluon.  
The dataset was from the Kaggle **IEEE-CIS Fraud Detection** competition.  
Due to Colab’s limited memory, I optimized the workflow for efficiency by using smaller data samples and chunked predictions.

---

## ⚙️ Steps Performed
1. Installed AutoGluon and Kaggle libraries.
2. Uploaded `kaggle.json` and authenticated with Kaggle.
3. Downloaded and extracted the **IEEE Fraud Detection** dataset.
4. Optimized memory usage by:
   - Dropping long text columns  
   - Sampling rows (`MAX_ROWS = 200,000`)
5. Trained a **LightGBM model** using:
   - Preset: `medium_quality_faster_train`
   - No bagging/stacking to save memory
6. Predicted probabilities in **chunks** to avoid runtime crashes.
7. Exported predictions as `my_submission.csv` for Kaggle submission.

---

## 🧠 Key Features
- RAM-optimized preprocessing  
- Single LightGBM model for efficiency  
- Safe chunked predictions for large datasets  
- Automated feature engineering and encoding  

---

## 🧾 Files in this Folder
- `AutoGluon_IEEE_Fraud.ipynb` – Low-RAM optimized Colab notebook  
- `Report.pdf` – Summary report with screenshots  

---

## 🏁 Results
AutoGluon successfully trained a fraud detection model that achieved a strong ROC-AUC score.  
The model handled millions of rows efficiently, and predictions were generated safely without memory crashes.
