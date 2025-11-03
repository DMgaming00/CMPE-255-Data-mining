# ⚖️ Binary Classification

This notebook applies **PyCaret’s classification module** to predict binary outcomes.

## 📘 Overview
- Dataset: *Titanic Dataset*
- Target: `Survived` (0 = No, 1 = Yes)
- Compared models automatically using `compare_models()`
- Evaluated the top model using confusion matrix and ROC curve

## ⚙️ Steps
1. Loaded and prepared the Titanic dataset
2. Ran `setup()` for binary classification
3. Compared multiple algorithms (Logistic Regression, Random Forest, etc.)
4. Finalized and saved the best model

## 📊 Outcome
- Predicted survival probabilities
- Generated `submission.csv` and `best_classification_model.pkl`
