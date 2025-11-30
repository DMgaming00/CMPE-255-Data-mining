# ---------------------------------------------------------------
# CMPE255 – Auto Data Toolkit (Streamlit Cloud – Safe Version)
# ---------------------------------------------------------------

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
    FunctionTransformer,
)
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    roc_curve, roc_auc_score
)
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.inspection import PartialDependenceDisplay

import shap

# ---------------------------------------------------------------
# Streamlit Layout Config
# ---------------------------------------------------------------
st.set_page_config(page_title="CMPE255 – Auto Data Toolkit", layout="wide", page_icon="📊")

st.title("🚀 CMPE255 – Auto Data Toolkit (Streamlit Cloud Safe Version)")

# ---------------------------------------------------------------
# Utility: Load Titanic
# ---------------------------------------------------------------
@st.cache_data
def load_titanic():
    try:
        return pd.read_csv("titanic.csv").drop(columns=["Name", "Ticket", "Cabin"], errors="ignore")
    except:
        st.error("Missing titanic.csv in repository. Upload it to the project root.")
        st.stop()

# ---------------------------------------------------------------
# Datetime Parser
# ---------------------------------------------------------------
def parse_datetimes(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                new = pd.to_datetime(df[col])
                if new.notna().sum() > 0:
                    df[col] = new
            except:
                pass
    # Add date features
    dt_cols = df.select_dtypes(include=["datetime64[ns]"]).columns
    for col in dt_cols:
        df[col+"_year"] = df[col].dt.year
        df[col+"_month"] = df[col].dt.month
        df[col+"_day"] = df[col].dt.day
    return df

# ---------------------------------------------------------------
# Outlier Removal (Streamlit-Safe)
# ---------------------------------------------------------------
def remove_outliers(df, method):
    if method == "None": return df
    df = df.copy()
    num = df.select_dtypes(include=["number"]).columns
    if len(num) == 0: return df

    if method == "IQR":
        for col in num:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            mask = (df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)
            df = df[mask]
        return df

    if method == "IsolationForest":
        iso = IsolationForest(contamination=0.03, random_state=42)
        preds = iso.fit_predict(df[num])
        return df[preds == 1]

    return df

# ---------------------------------------------------------------
# Sidebar – Controls
# ---------------------------------------------------------------
with st.sidebar:
    st.header("1. Data Source")
    src = st.radio("Choose:", ["Titanic demo", "Upload CSV"])

    file = None
    if src == "Upload CSV":
        file = st.file_uploader("Upload CSV", type=["csv"])

    st.header("2. Target")
    tcol_input = st.text_input("Target Column", value="Survived")

    st.header("3. Train/Test")
    test_size = st.slider("Test size", 0.1, 0.4, 0.2)

    st.header("4. Cleaning")
    imputer_method = st.selectbox("Imputation", ["Mean", "Median", "KNN", "Iterative"])
    outlier_method = st.selectbox("Outliers", ["None", "IQR", "IsolationForest"])
    skew_method = st.selectbox("Fix Skewness", ["None", "Log", "Yeo-Johnson"])

    st.header("5. Encoding / FS")
    encoding_method = st.selectbox("Encoding", ["OneHot", "Ordinal"])
    fs_method = st.selectbox("Feature selection", ["None", "Variance Threshold", "RFE"])

    run = st.button("🚀 Run Auto Toolkit")

# ---------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------
if src == "Titanic demo":
    df_raw = load_titanic()
else:
    if file is None:
        st.info("Upload a CSV to proceed.")
        st.stop()
    df_raw = pd.read_csv(file)

st.subheader("📊 Raw Data")
st.write(df_raw.head())

# Clean duplicates
before = len(df_raw)
df = df_raw.drop_duplicates()
after = len(df)
st.write(f"Removed {before - after} duplicate rows.")

# Parse date features
df = parse_datetimes(df)

# Remove outliers
df = remove_outliers(df, outlier_method)

# ---------------------------------------------------------------
# Target Selection
# ---------------------------------------------------------------
target = tcol_input if tcol_input in df.columns else st.selectbox("Select target:", df.columns)

if target not in df.columns:
    st.error("Invalid target column.")
    st.stop()

X = df.drop(columns=[target])
y = df[target]

if y.dtype == "object":
    y = y.astype("category").cat.codes

# ---------------------------------------------------------------
# Pipeline Builders
# ---------------------------------------------------------------
def get_imputer():
    if imputer_method == "Mean": return SimpleImputer(strategy="mean")
    if imputer_method == "Median": return SimpleImputer(strategy="median")
    if imputer_method == "KNN": return KNNImputer(n_neighbors=5)
    return IterativeImputer(random_state=42)

def get_skew_tf():
    if skew_method == "Log":
        return FunctionTransformer(lambda x: np.log1p(np.clip(x, a_min=0.0001, a_max=None)))
    if skew_method == "Yeo-Johnson":
        from sklearn.preprocessing import PowerTransformer
        return PowerTransformer(method="yeo-johnson")
    return "passthrough"

def get_encoder():
    if encoding_method == "Ordinal":
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    return OneHotEncoder(handle_unknown="ignore")

# Build preprocessing
num_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

num_pipe = Pipeline([
    ("imputer", get_imputer()),
    ("skew", get_skew_tf()),
    ("scale", StandardScaler())
])

cat_pipe = Pipeline([
    ("enc", get_encoder())
])

pre = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])

# Optional FS
if fs_method == "Variance Threshold":
    fs_step = VarianceThreshold(0.0)
elif fs_method == "RFE":
    base = RandomForestClassifier(n_estimators=100, random_state=42)
    fs_step = RFE(base, n_features_to_select=10)
else:
    fs_step = "passthrough"

pipe = Pipeline([
    ("pre", pre),
    ("fs", fs_step),
    ("model", RandomForestClassifier(n_estimators=300, random_state=42))
])

# ---------------------------------------------------------------
# Run Pipeline
# ---------------------------------------------------------------
if not run:
    st.stop()

with st.spinner("Training..."):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    pipe.fit(X_train, y_train)

st.success("Model training complete.")

# ---------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------
y_pred = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

st.subheader("📈 Metrics")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Accuracy", f"{acc:.3f}")
c2.metric("Precision", f"{prec:.3f}")
c3.metric("Recall", f"{rec:.3f}")
c4.metric("F1-score", f"{f1:.3f}")

st.code(classification_report(y_test, y_pred), language="text")

# ---------------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------------
st.subheader("📊 Confusion Matrix")
fig_cm, ax = plt.subplots(figsize=(4,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig_cm)

# ---------------------------------------------------------------
# ROC Curve
# ---------------------------------------------------------------
st.subheader("📉 ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)
fig_roc, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"AUC={auc:.3f}")
ax.plot([0,1],[0,1],"k--")
ax.legend()
st.pyplot(fig_roc)

# ---------------------------------------------------------------
# Feature Importance (RF)
# ---------------------------------------------------------------
st.subheader("📊 Feature Importance (RandomForest)")
rf = pipe.named_steps["model"]
importances = rf.feature_importances_

# Get feature names
def get_names(pre):
    names = []
    for name, trans, cols in pre.transformers_:
        if name == "num":
            names.extend(cols)
        elif name == "cat":
            enc = trans.named_steps["enc"]
            if hasattr(enc, "get_feature_names_out"):
                names.extend(enc.get_feature_names_out(cols))
            else:
                names.extend(cols)
    return names

feat_names = get_names(pipe.named_steps["pre"])
fi_df = pd.DataFrame({"feat": feat_names, "imp": importances}).sort_values("imp", ascending=False)

fig_fi, ax_fi = plt.subplots(figsize=(6,5))
sns.barplot(data=fi_df.head(10), y="feat", x="imp", ax=ax_fi)
st.pyplot(fig_fi)

# ---------------------------------------------------------------
# PDP (Safe)
# ---------------------------------------------------------------
st.subheader("📉 Partial Dependence Plots")
pdp_cols = [c for c in ["Age","Fare","Pclass"] if c in X.columns]
cols = st.columns(len(pdp_cols))

for col, feat in zip(cols, pdp_cols):
    fig, ax = plt.subplots()
    PartialDependenceDisplay.from_estimator(pipe, X, [feat], ax=ax)
    col.pyplot(fig)

# ---------------------------------------------------------------
# SHAP (STREAMLIT-CLOUD SAFE)
# ---------------------------------------------------------------
st.subheader("🧠 SHAP Feature Importance (Custom Beeswarm)")

try:
    sample = X_train.sample(min(200, len(X_train)), random_state=42)
    Xs_proc = pipe.named_steps["pre"].transform(sample)
    explainer = shap.TreeExplainer(rf)
    sv_all = explainer.shap_values(Xs_proc)

    # Handle binary list case
    if isinstance(sv_all, list):
        sv = sv_all[1]
    else:
        sv = sv_all

    # Custom beeswarm
    fig, ax = plt.subplots(figsize=(8,6))
    shap.summary_plot(
        sv, features=Xs_proc, feature_names=feat_names,
        show=False, plot_type="dot", max_display=10
    )
    st.pyplot(fig)
except Exception as e:
    st.error(f"SHAP failed: {e}")

# ---------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------
st.subheader("📥 Download Cleaned Dataset")
st.download_button(
    "Download CSV",
    df.to_csv(index=False).encode(),
    file_name="cleaned_dataset.csv",
    mime="text/csv"
)

# Report
html = f"""
<h1>Model Report</h1>
<p>Accuracy:{acc:.3f}</p>
<p>Precision:{prec:.3f}</p>
<p>Recall:{rec:.3f}</p>
<p>F1-score:{f1:.3f}</p>
<pre>{classification_report(y_test, y_pred)}</pre>
"""
st.download_button(
    "Download HTML Report",
    html.encode(),
    file_name="model_report.html",
    mime="text/html"
)
