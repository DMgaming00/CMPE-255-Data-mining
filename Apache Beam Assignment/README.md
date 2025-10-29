# 🚀 Apache Beam Data Engineering Exercise

This project demonstrates core **Apache Beam** features using a **Google Colab notebook** for CMPE-255 (Data Mining).  
It covers everything from simple transforms to machine learning inference — all inside a single Beam pipeline.

---

## 📘 Overview

The notebook showcases how to use Apache Beam for batch and streaming-style data processing.  
It walks through multiple Beam concepts including:

- **Pipeline creation**
- **Map / Filter transforms**
- **ParDo (custom DoFn) and Composite transforms**
- **Windowing** with time-based aggregation
- **Partitioning** data into multiple outputs
- (Optional) **Machine Learning inference** using Beam ML and scikit-learn

---

## 🧱 Project Structure

```
apache_beam_assignment_colab_PATCHED_v2.ipynb   # Main Colab notebook
beam_outputs/                                   # Output files from each pipeline stage
├── big_items-00000-of-00001
├── composite_result-00000-of-00001
├── heavy-00000-of-00001
├── light-00000-of-00001
├── windowed_totals-00000-of-00001
└── ml_predictions-00000-of-00001 (optional)
```

---

## ⚙️ How to Run

### 1. Open in Google Colab
- Upload the notebook [`apache_beam_assignment_colab_PATCHED_v2.ipynb`](https://colab.research.google.com/drive/1JKGgPiBDLUH8HCe4UGITRAs_DLXwtBN7)
- Or open directly in Colab via  
  **File → Upload notebook → Choose file**

### 2. Install Dependencies
Run the first cell to install Apache Beam and scikit-learn.  
If Colab asks to restart the runtime, do so before running the rest of the notebook.

### 3. Run All Cells
Execute cells in order (Runtime → Run all).  
Beam will automatically create the folder `beam_outputs/` and write text files with results.

### 4. Check Outputs
In Colab’s left sidebar (Files tab), expand the **beam_outputs/** folder to view results.

---

## 🧾 What Each Output Means

| Output File | Description |
|--------------|-------------|
| `big_items` | Result of Map + Filter — only fruits with quantity ≥ 5 |
| `composite_result` | Custom ParDo + Composite transform (uppercase + tagging) |
| `windowed_totals` | Windowing with 1-minute aggregation |
| `heavy` | Partitioned data — heavy items (qty ≥ 5) |
| `light` | Partitioned data — light items (qty < 5) |
| `ml_predictions` | (Optional) ML inference results from a logistic regression model |

---

## 🗣️ Suggested Video Walkthrough

If you’re submitting this for class:
1. Show the notebook structure and run each major section.
2. Open each output file and explain what it represents.
3. End with a quick summary of how Beam handles transforms, windows, and parallel processing.

Example closing line:
> “This project demonstrates Apache Beam’s flexibility for both batch and streaming data processing, and how it integrates with machine learning pipelines.”

---

## 🧰 Tools Used

- **Python 3.12**
- **Apache Beam**
- **Google Colab**
- **scikit-learn**
- **NumPy / joblib**

---

## 🏁 Credits

**Author:** *Dev Mulchandani*  
**Course:** CMPE 255 – Data Mining, Fall 2025  
**Instructor:** *Vijay Eranti*  
**University:** San José State University

---

## 📄 License

This project is for educational purposes only.  
Feel free to fork, modify, and experiment for your own learning.

---
