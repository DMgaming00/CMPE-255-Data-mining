# 🧠 AutoGluon Assignment
### Submitted by: **Dev Mulchandani**

This repository contains all three parts of my **AutoGluon assignment**, where I implemented and demonstrated end-to-end machine learning workflows using **AutoGluon’s TabularPredictor**.  
Each section includes a Colab notebook, report, and sample outputs.

---

## 📁 Repository Structure

```
AutoGluon-Assignment/
│
├── 🏡 California_House_Prices/
│   ├── AutoGluon_House_Prices.ipynb
│   ├── Report.pdf
│   └── README.txt
│
├── 💳 IEEE_Fraud_Detection/
│   ├── AutoGluon_IEEE_Fraud.ipynb
│   ├── Report.pdf
│   └── README.txt
│
├── ⚙️ Part_2_Demos/
│   ├── AutoGluon_Part2_QuickDemo.ipynb
│   ├── Report.pdf
│   └── README.txt
│
└── README.md
```

---

## 🏡 Part 1(a): California House Prices
**Goal:** Predict house sale prices using tabular regression.  
- Implemented using the *California House Prices Kaggle dataset*.  
- Used AutoGluon’s `TabularPredictor` for regression with `medium_quality` presets.  
- The model automatically handled preprocessing, feature selection, and ensembling.  
- Final predictions were saved as `my_submission.csv` ready for Kaggle submission.  

📄 *Files:*  
- `AutoGluon_House_Prices.ipynb` – Colab notebook  
- `Report.pdf` – Summary and screenshots  

---

## 💳 Part 1(b): IEEE Fraud Detection
**Goal:** Detect fraudulent transactions efficiently in a large Kaggle dataset.  
- Connected to Kaggle API for automated dataset download.  
- Optimized for Colab’s RAM by dropping long text columns and sampling rows.  
- Trained a LightGBM model using AutoGluon’s TabularPredictor.  
- Used chunked prediction to generate probability-based outputs safely without crashes.  
- Final submission saved as `my_submission.csv`.

📄 *Files:*  
- `AutoGluon_IEEE_Fraud.ipynb` – Low-RAM optimized Colab notebook  
- `Report.pdf` – Summary with screenshots and explanation  

---

## ⚙️ Part 2: AutoGluon Demonstrations
**Goal:** Showcase AutoGluon’s versatility using small, fast demos.  
Includes:  
1️⃣ **Classification** — Adult Income dataset from OpenML (predict income > $50K).  
2️⃣ **Regression** — California Housing dataset (predict median home value).  
3️⃣ **Multimodal Tabular** — Combines numeric + text features to predict spending behavior.  
4️⃣ **Automatic Feature Engineering** — Shows AutoGluon’s built-in preprocessing and feature importance tools.  

Each demo runs quickly in Colab and demonstrates how AutoGluon handles different ML tasks with minimal code.

📄 *Files:*  
- `AutoGluon_Part2_QuickDemo.ipynb` – Colab notebook  
- `Report.pdf` – Explanation and screenshots  

---

## 🧩 Tools & Libraries
- **AutoGluon** – Automated Machine Learning (AutoML) toolkit  
- **Kaggle API** – For dataset access (Part 1)  
- **scikit-learn** – Used for OpenML and California Housing datasets  
- **Google Colab** – Execution environment  

---

## 🚀 How to Run
1. Open any `.ipynb` file in **Google Colab**.  
2. Run the setup cells to install dependencies (`!pip install autogluon`).  
3. For Kaggle notebooks, upload your `kaggle.json` key.  
4. Run all cells sequentially — outputs and results will appear inline.  

---

## 🏁 Summary
This project demonstrates how AutoGluon simplifies complex machine-learning tasks.  
It automatically handles:
- Data preprocessing  
- Feature engineering  
- Model selection  
- Training and evaluation  
- Prediction and export  

All with minimal coding effort and strong performance across classification, regression, and multimodal data problems.
