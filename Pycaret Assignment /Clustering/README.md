# 🧮 Clustering

This notebook uses **PyCaret’s clustering module** to group similar data points automatically.

## 📘 Overview
- Algorithm: **K-Means**
- Automatically handled normalization and scaling
- Visualized clusters using a 2D scatter plot

## ⚙️ Steps
1. Loaded and cleaned dataset
2. Initialized PyCaret with `setup()`
3. Created a K-Means model using `create_model('kmeans')`
4. Assigned clusters using `assign_model()`
5. Visualized clusters and saved results

## 📊 Outcome
- Grouped data into meaningful clusters
- Exported `best_clustering_model.pkl` and labeled dataset
