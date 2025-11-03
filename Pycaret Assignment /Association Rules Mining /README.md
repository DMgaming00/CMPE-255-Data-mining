# 🧩 Association Rules Mining

This notebook performs **Market Basket Analysis** using the **Apriori algorithm** from `mlxtend`.

## 📘 Overview
- Dataset: *Bread Basket Dataset*
- One-hot encoded transactions for Apriori
- Generated **frequent itemsets** and **association rules**
- Evaluated rules based on **support**, **confidence**, and **lift**

## ⚙️ Steps
1. Loaded and explored transactional data
2. Transformed data into basket (0/1) format
3. Applied `apriori()` and `association_rules()`
4. Filtered and saved the strongest rules

## 📊 Outcome
- Discovered relationships between frequently purchased items
- Exported top rules to `association_rules.csv`
