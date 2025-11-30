import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    classification_report,
)
from sklearn.inspection import partial_dependence

import shap

# --------------------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="CMPE 255 – Auto Data Toolkit",
    layout="wide",
    page_icon="🚢",
)

st.title("🚢 CMPE 255 – Auto Data Toolkit")
st.markdown(
    '''
This Streamlit app demonstrates a **CRISP-DM style** workflow:

1. Choose a dataset (Titanic demo or upload CSV)  
2. Select the target column  
3. Automatically clean & preprocess the data  
4. Train a RandomForest model inside a scikit-learn Pipeline  
5. See metrics, confusion matrix, ROC curve  
6. Explore **feature importance** and **SHAP explainability**  
7. Look at **Partial Dependence Plots (PDP)** for key numeric features  
'''
)

# --------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------
@st.cache_data
def load_titanic_demo() -> pd.DataFrame:
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    # Drop very wide or mostly-missing columns that cause unstable UI / poor modeling
    df = df.drop(columns=["Name", "Ticket", "Cabin"], errors="ignore")
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    num_tf = Pipeline([("scaler", StandardScaler())])
    cat_tf = Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore"))])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_tf, numeric_cols),
            ("cat", cat_tf, categorical_cols),
        ]
    )
    return pre, numeric_cols, categorical_cols


def run_pipeline(df: pd.DataFrame, target_col: str, test_size: float = 0.2):
    assert target_col in df.columns, f"Target column '{target_col}' not found in data."

    # Basic cleaning: drop rows with missing target
    df = df.dropna(subset=[target_col]).copy()

    y = df[target_col]
    X = df.drop(columns=[target_col])

    pre, num_cols, cat_cols = build_preprocessor(X)

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline([("pre", pre), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
    }

    # ROC / AUC: only if binary and predict_proba available
    roc_info = None
    y_unique = y.unique()
    if hasattr(pipe, "predict_proba") and len(y_unique) == 2:
        try:
            y_proba = pipe.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            roc_info = {"fpr": fpr, "tpr": tpr, "auc": float(auc)}
        except Exception:
            roc_info = None
    else:
        y_proba = None

    return {
        "pipeline": pipe,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "metrics": metrics,
        "roc": roc_info,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
    }


# --------------------------------------------------------------------------------------
# Sidebar: data selection and target choice
# --------------------------------------------------------------------------------------
st.sidebar.header("1. Choose Data Source")

data_source = st.sidebar.radio(
    "Data source:",
    ["Titanic demo", "Upload CSV"],
    index=0,
)

if data_source == "Titanic demo":
    df = load_titanic_demo()
    st.sidebar.success("Loaded Titanic demo dataset.")
else:
    uploaded = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded is None:
        st.info("👈 Upload a CSV file in the sidebar to begin.")
        st.stop()
    df = pd.read_csv(uploaded)
    st.sidebar.success("Custom CSV loaded.")

st.subheader("📊 Raw Data Preview")
st.write(f"Shape: **{df.shape[0]} rows × {df.shape[1]} columns**")
st.dataframe(df.head(), use_container_width=True)

st.subheader("🧼 Missing Values")
missing = df.isna().mean().to_frame("missing_fraction").T
st.dataframe(missing, use_container_width=True)

# Guess likely target columns
default_target_candidates = [c for c in df.columns if c.lower() in ["survived", "target", "label", "class", "outcome", "y"]]
if default_target_candidates:
    default_target = default_target_candidates[0]
else:
    default_target = df.columns[-1]

st.sidebar.header("2. Target Column")
target_col = st.sidebar.selectbox(
    "Select the target column:",
    df.columns.tolist(),
    index=df.columns.get_loc(default_target),
)

st.sidebar.header("3. Train / Test Split")
test_size = st.sidebar.slider("Test size fraction", 0.1, 0.4, 0.2, step=0.05)

run_button = st.sidebar.button("🚀 Run Auto-Toolkit")

if not run_button:
    st.info("👈 Configure options in the sidebar and click **Run Auto-Toolkit**.")
    st.stop()

# --------------------------------------------------------------------------------------
# Run pipeline
# --------------------------------------------------------------------------------------
with st.spinner("Training model and computing metrics..."):
    results = run_pipeline(df, target_col=target_col, test_size=test_size)

pipe = results["pipeline"]
X = results["X"]
y = results["y"]
X_train = results["X_train"]
X_test = results["X_test"]
y_train = results["y_train"]
y_test = results["y_test"]
metrics = results["metrics"]
roc_info = results["roc"]
num_cols = results["num_cols"]
cat_cols = results["cat_cols"]

st.success("✅ Model training complete.")

# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
st.subheader("📈 Evaluation Metrics")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
col2.metric("Precision (weighted)", f"{metrics['precision_weighted']:.3f}")
col3.metric("Recall (weighted)", f"{metrics['recall_weighted']:.3f}")
col4.metric("F1 (weighted)", f"{metrics['f1_weighted']:.3f}")

st.text("Classification report:")
st.text(classification_report(y_test, pipe.predict(X_test)))

# --------------------------------------------------------------------------------------
# Confusion Matrix
# --------------------------------------------------------------------------------------
st.subheader("🧮 Confusion Matrix")
cm = confusion_matrix(y_test, pipe.predict(X_test))

fig_cm, ax_cm = plt.subplots()
im = ax_cm.imshow(cm, cmap="Blues")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax_cm.text(j, i, cm[i, j], ha="center", va="center", color="black")
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("True")
fig_cm.colorbar(im, ax=ax_cm)
st.pyplot(fig_cm)

# --------------------------------------------------------------------------------------
# ROC Curve (if available)
# --------------------------------------------------------------------------------------
st.subheader("📉 ROC Curve")
if roc_info is not None:
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(roc_info["fpr"], roc_info["tpr"], label=f"AUC = {roc_info['auc']:.3f}")
    ax_roc.plot([0, 1], [0, 1], "k--", label="Random")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend()
    st.pyplot(fig_roc)
else:
    st.info("ROC curve is only shown for binary classification problems with predict_proba.")

# --------------------------------------------------------------------------------------
# Global Feature Importance (model-based)
# --------------------------------------------------------------------------------------
st.subheader("📊 Feature Importance (RandomForest)")

X_train_proc = pipe["pre"].transform(X_train)
if hasattr(X_train_proc, "toarray"):
    X_train_proc = X_train_proc.toarray()

feature_names = []
if num_cols:
    feature_names.extend(num_cols)
if cat_cols:
    ohe = pipe["pre"].named_transformers_["cat"]["encoder"]
    cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
    feature_names.extend(cat_feature_names)

importances = pipe["model"].feature_importances_

fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
fi_df = fi_df.sort_values("importance", ascending=False)

fig_fi, ax_fi = plt.subplots(figsize=(6, max(3, len(fi_df) * 0.3)))
ax_fi.barh(fi_df["feature"], fi_df["importance"])
ax_fi.invert_yaxis()
ax_fi.set_xlabel("Importance")
ax_fi.set_title("RandomForest Feature Importance")
st.pyplot(fig_fi)

# --------------------------------------------------------------------------------------
# SHAP Feature Importance
# --------------------------------------------------------------------------------------
st.subheader("🧠 SHAP Feature Importance")

try:
    # Sample for speed
    sample_size = min(300, len(X_train))
    X_sample = X_train.sample(sample_size, random_state=42)

    X_sample_proc = pipe["pre"].transform(X_sample)
    if hasattr(X_sample_proc, "toarray"):
        X_sample_proc = X_sample_proc.toarray()

    X_sample_df = pd.DataFrame(X_sample_proc, columns=feature_names)

    explainer = shap.TreeExplainer(pipe["model"])
    shap_values = explainer.shap_values(X_sample_df)

    # Handle binary vs multiclass
    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            # Binary: take class 1
            sv = shap_values[1]
        else:
            # Multiclass: average absolute SHAP over classes
            sv_arr = np.array(shap_values)  # (n_classes, n_samples, n_features)
            sv = sv_arr.mean(axis=0)
    else:
        sv = shap_values

    fig_shap = plt.figure(figsize=(7, 5))
    shap.summary_plot(sv, X_sample_df, feature_names=feature_names, show=False)
    st.pyplot(fig_shap)

except Exception as e:
    st.warning(f"SHAP failed: {e}")

# --------------------------------------------------------------------------------------
# Partial Dependence Plots (PDP)
# --------------------------------------------------------------------------------------
st.subheader("📉 Partial Dependence Plots (PDP)")

numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
# Try some typical interesting numeric columns:
candidate_pdp_cols = [c for c in ["Age", "Fare", "Pclass"] if c in numeric_cols]
if not candidate_pdp_cols:
    candidate_pdp_cols = numeric_cols[:2]

if not candidate_pdp_cols:
    st.info("No numeric columns found to plot PDP.")
else:
    cols = st.columns(len(candidate_pdp_cols))
    for ax_col, feature in zip(cols, candidate_pdp_cols):
        with ax_col:
            try:
                pdp_res = partial_dependence(pipe, X=X_train, features=[feature])
                grid = pdp_res.get("grid_values", pdp_res.get("values"))[0]
                avg = pdp_res["average"][0]

                fig_pdp, ax_pdp = plt.subplots()
                ax_pdp.plot(grid, avg)
                ax_pdp.set_xlabel(feature)
                ax_pdp.set_ylabel("Partial dependence")
                ax_pdp.set_title(f"PDP – {feature}")
                st.pyplot(fig_pdp)
            except Exception as e:
                st.warning(f"Could not compute PDP for {feature}: {e}")
