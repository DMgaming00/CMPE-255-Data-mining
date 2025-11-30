# app.py
# --------------------------------------------------------
# CMPE 255 – Auto Data Toolkit (Streamlit App)
# --------------------------------------------------------
# Requirements (add to requirements.txt):
# streamlit
# pandas
# numpy
# scikit-learn
# matplotlib
# seaborn
# shap
# ydata-profiling
# category_encoders
# --------------------------------------------------------

import io
import textwrap
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

from sklearn.impute import (
    SimpleImputer,
    KNNImputer,
)
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)
from sklearn.feature_selection import VarianceThreshold, RFE, mutual_info_classif
from sklearn.inspection import PartialDependenceDisplay

import shap

# --------------------------------------------------------
# Streamlit page config
# --------------------------------------------------------
st.set_page_config(
    page_title="CMPE 255 – Auto Data Toolkit",
    layout="wide",
    page_icon="📊",
)

st.title("🚀 CMPE 255 – Auto Data Toolkit")

# --------------------------------------------------------
# Utility functions
# --------------------------------------------------------
@st.cache_data
def load_titanic_demo() -> pd.DataFrame:
    """
    Load Titanic demo dataset from a local CSV.
    Put your titanic.csv in the repo root or adjust the path here.
    """
    try:
        df = pd.read_csv("titanic.csv")
    except FileNotFoundError:
        st.error(
            "titanic.csv not found. Please add titanic.csv to the repo or adjust the path in app.py."
        )
        st.stop()
    return df


def try_parse_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to parse object columns as datetime and create year/month/day features.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            # Try parsing; if it fails, keep original
            try:
                parsed = pd.to_datetime(df[col], errors="raise")
                # If this worked and has at least some non-NaT values, keep it
                if parsed.notna().sum() > 0:
                    df[col] = parsed
            except Exception:
                pass

    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns
    for col in datetime_cols:
        df[f"{col}_year"] = df[col].dt.year
        df[f"{col}_month"] = df[col].dt.month
        df[f"{col}_day"] = df[col].dt.day

    return df


def remove_outliers(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """
    Remove outliers using IQR / Z-score / IsolationForest / None.
    Applied only to numeric columns.
    """
    if method == "None":
        return df

    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=["number"]).columns
    if len(numeric_cols) == 0:
        return df_clean

    if method == "IQR":
        for col in numeric_cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            mask = (df_clean[col] >= Q1 - 1.5 * IQR) & (df_clean[col] <= Q3 + 1.5 * IQR)
            df_clean = df_clean[mask]

    elif method == "Z-score":
        from scipy.stats import zscore

        z = np.abs(zscore(df_clean[numeric_cols]))
        df_clean = df_clean[(z < 3).all(axis=1)]

    elif method == "IsolationForest":
        iso = IsolationForest(contamination=0.02, random_state=42)
        preds = iso.fit_predict(df_clean[numeric_cols])
        df_clean = df_clean[preds == 1]

    return df_clean


def make_imputer(method: str):
    if method == "Mean":
        return SimpleImputer(strategy="mean")
    elif method == "Median":
        return SimpleImputer(strategy="median")
    elif method == "KNN Imputer":
        return KNNImputer(n_neighbors=5)
    elif method == "Iterative Imputer":
        return IterativeImputer(random_state=42)
    else:
        # Pass-through – but SimpleImputer still needed to keep pipeline happy
        return SimpleImputer(strategy="most_frequent")


def make_skew_transformer(method: str):
    # Use FunctionTransformer to be safe with non-positive values
    if method == "Log":

        def log1p_safe(x):
            x = np.array(x, dtype=float)
            x[x <= 0] = np.nan
            return np.log1p(x)

        return FunctionTransformer(log1p_safe, validate=False)

    elif method in ("Box-Cox", "Yeo-Johnson"):
        from sklearn.preprocessing import PowerTransformer

        if method == "Box-Cox":
            return PowerTransformer(method="box-cox")  # requires positive
        else:
            return PowerTransformer(method="yeo-johnson")
    else:
        return "passthrough"


def make_encoder(method: str):
    if method == "OneHot":
        return OneHotEncoder(handle_unknown="ignore")
    elif method == "Ordinal":
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    else:
        # Target Encoding via category_encoders
        try:
            import category_encoders as ce

            return ce.TargetEncoder()
        except Exception:
            st.warning(
                "category_encoders not installed – falling back to OneHotEncoder."
            )
            return OneHotEncoder(handle_unknown="ignore")


def build_preprocessor(
    X: pd.DataFrame,
    imputer_method: str,
    skew_method: str,
    encoding_method: str,
):
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    num_imputer = make_imputer(imputer_method)
    skew_tf = make_skew_transformer(skew_method)
    cat_encoder = make_encoder(encoding_method)

    num_pipeline_steps = [("imputer", num_imputer)]
    if skew_tf != "passthrough":
        num_pipeline_steps.append(("skew", skew_tf))
    num_pipeline_steps.append(("scaler", StandardScaler()))

    num_pipeline = Pipeline(num_pipeline_steps)
    cat_pipeline = Pipeline([("encoder", cat_encoder)])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_cols),
            ("cat", cat_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor, numeric_cols, categorical_cols


# --------------------------------------------------------
# Sidebar – Controls
# --------------------------------------------------------
with st.sidebar:
    st.header("1. Choose Data Source")
    ds_choice = st.radio(
        "Data source:",
        ["Titanic demo", "Upload CSV"],
        index=0,
    )

    if ds_choice == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    else:
        uploaded = None

    st.header("2. Target Column / Split")
    target_col_manual = st.text_input(
        "Target column name (leave blank to auto-guess 'Survived'):",
        value="Survived",
    )

    test_size = st.slider("Test size fraction", 0.1, 0.4, 0.2, 0.05, help="Hold-out size")

    st.header("3. Data Cleaning Options")

    imputer_method = st.selectbox(
        "Missing value imputation",
        ["Mean", "Median", "KNN Imputer", "Iterative Imputer"],
        index=0,
    )

    outlier_method = st.selectbox(
        "Outlier detection / removal",
        ["None", "IQR", "Z-score", "IsolationForest"],
        index=0,
    )

    skew_method = st.selectbox(
        "Skewness transform (numeric)",
        ["None", "Log", "Box-Cox", "Yeo-Johnson"],
        index=0,
    )

    encoding_method = st.selectbox(
        "Categorical encoding",
        ["OneHot", "Ordinal", "Target Encoding"],
        index=0,
    )

    fs_method = st.selectbox(
        "Feature selection",
        ["None", "Variance Threshold", "RFE (RandomForest)"],
        index=0,
        help="Mutual Information is computed for visualization only.",
    )

    st.header("4. Extra")
    generate_profile = st.checkbox("Generate data profiling report (ydata-profiling)")
    run_button = st.button("✨ Run Auto-Toolkit", type="primary")

# --------------------------------------------------------
# Load data
# --------------------------------------------------------
if ds_choice == "Titanic demo":
    df_raw = load_titanic_demo()
else:
    if uploaded is None:
        st.info("Upload a CSV file to begin.")
        st.stop()
    else:
        df_raw = pd.read_csv(uploaded)

st.subheader("📊 Raw Data Preview")
st.write(f"Shape: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
st.dataframe(df_raw.head())

# --------------------------------------------------------
# Pre-cleaning stats
# --------------------------------------------------------
st.subheader("🧼 Missing Values")
missing_fraction = df_raw.isna().mean().to_frame("missing_fraction").T
st.dataframe(missing_fraction)

# Duplicates
n_before = len(df_raw)
df = df_raw.drop_duplicates()
n_after = len(df)
st.write(f"Removed **{n_before - n_after}** duplicate rows.")

# Datetime parsing & feature engineering
df = try_parse_datetimes(df)

# Outlier removal
df = remove_outliers(df, outlier_method)

# --------------------------------------------------------
# Choose target column
# --------------------------------------------------------
if target_col_manual and target_col_manual in df.columns:
    target_col = target_col_manual
elif "Survived" in df.columns:
    target_col = "Survived"
else:
    target_col = st.selectbox("Select target column", df.columns)

if target_col not in df.columns:
    st.error("Target column not found in dataset.")
    st.stop()

X = df.drop(columns=[target_col])
y = df[target_col]

# For binary classification, map yes/no etc. to 0/1 when necessary
if y.dtype == "object":
    y = y.astype("category").cat.codes

# --------------------------------------------------------
# Optional profiling report
# --------------------------------------------------------
if generate_profile:
    st.subheader("📋 Profiling Report")
    try:
        from ydata_profiling import ProfileReport

        profile = ProfileReport(df, title="Data Profile", explorative=True)
        buffer = io.StringIO()
        profile.to_file("profile_report.html")
        with open("profile_report.html", "rb") as f:
            st.download_button(
                "Download profile_report.html", f, file_name="profile_report.html"
            )
        st.success("Profiling report generated.")
    except Exception as e:
        st.warning(f"Could not generate profile report: {e}")

# --------------------------------------------------------
# Train / Test split
# --------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)

# --------------------------------------------------------
# Build preprocessing and pipeline
# --------------------------------------------------------
preprocessor, num_cols, cat_cols = build_preprocessor(
    X, imputer_method, skew_method, encoding_method
)

# Feature selection step
if fs_method == "Variance Threshold":
    fs_step = VarianceThreshold(threshold=0.0)
elif fs_method.startswith("RFE"):
    base_est = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    )
    fs_step = RFE(base_est, n_features_to_select=10)
else:
    fs_step = "passthrough"

prep_and_fs = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("fs", fs_step),
    ]
)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
)

full_pipeline = Pipeline(
    steps=[
        ("prep_fs", prep_and_fs),
        ("model", model),
    ]
)

# --------------------------------------------------------
# Run Auto-Toolkit
# --------------------------------------------------------
if not run_button:
    st.stop()

with st.spinner("Training model and running toolkit..."):
    full_pipeline.fit(X_train, y_train)

st.success("✅ Model training complete.")

# --------------------------------------------------------
# Evaluation metrics
# --------------------------------------------------------
y_pred = full_pipeline.predict(X_test)
y_proba = full_pipeline.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

st.subheader("📈 Evaluation Metrics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{acc:.3f}")
c2.metric("Precision (weighted)", f"{prec:.3f}")
c3.metric("Recall (weighted)", f"{rec:.3f}")
c4.metric("F1-score (weighted)", f"{f1:.3f}")

st.markdown("#### Classification report")
report_text = classification_report(y_test, y_pred, digits=3)
st.text(report_text)

# Confusion Matrix
st.subheader("📊 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    ax=ax_cm,
)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("True")
st.pyplot(fig_cm)

# ROC Curve
st.subheader("📉 ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)
fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
ax_roc.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
ax_roc.plot([0, 1], [0, 1], "k--", label="Random")
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.legend()
st.pyplot(fig_roc)

# --------------------------------------------------------
# Feature importance (RandomForest)
# --------------------------------------------------------
st.subheader("📊 Feature Importance (RandomForest)")

# Get processed feature names from ColumnTransformer
def get_feature_names(preprocessor: ColumnTransformer) -> list:
    output_features = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(trans, "named_steps"):
            last_step = list(trans.named_steps.values())[-1]
        else:
            last_step = trans

        if hasattr(last_step, "get_feature_names_out"):
            feats = last_step.get_feature_names_out(cols)
        elif hasattr(last_step, "get_feature_names"):
            feats = last_step.get_feature_names(cols)
        else:
            feats = cols
        output_features.extend(feats)
    return list(output_features)


prep_fs = full_pipeline.named_steps["prep_fs"]
preprocessor_fitted = prep_fs.named_steps["preprocess"]

try:
    feature_names = get_feature_names(preprocessor_fitted)
except Exception:
    feature_names = [f"f{i}" for i in range(prep_fs.transform(X_train[:1]).shape[1])]

rf_model = full_pipeline.named_steps["model"]
importances = rf_model.feature_importances_
fi_df = (
    pd.DataFrame({"feature": feature_names, "importance": importances})
    .sort_values("importance", ascending=False)
)

fig_fi, ax_fi = plt.subplots(figsize=(6, 4))
sns.barplot(
    data=fi_df.head(15),
    y="feature",
    x="importance",
    ax=ax_fi,
)
ax_fi.set_title("RandomForest Feature Importance (Top 15)")
st.pyplot(fig_fi)

# Mutual Information (for reference – not used in pipeline)
st.subheader("📊 Mutual Information (top 15 – not used in pipeline)")
try:
    X_proc = prep_fs.transform(X_train)
    mi = mutual_info_classif(X_proc, y_train, discrete_features=False, random_state=42)
    mi_df = (
        pd.DataFrame({"feature": feature_names, "mi": mi})
        .sort_values("mi", ascending=False)
    )
    fig_mi, ax_mi = plt.subplots(figsize=(6, 4))
    sns.barplot(data=mi_df.head(15), y="feature", x="mi", ax=ax_mi)
    ax_mi.set_title("Mutual Information (Top 15)")
    st.pyplot(fig_mi)
except Exception as e:
    st.info(f"Could not compute mutual information: {e}")

# --------------------------------------------------------
# Partial Dependence Plots (PDP)
# --------------------------------------------------------
st.subheader("📉 Partial Dependence Plots (PDP)")
pdp_cols = []
for col in ["Age", "Fare", "Pclass"]:
    if col in X.columns:
        pdp_cols.append(col)

if len(pdp_cols) == 0:
    st.info("No standard numeric columns (Age/Fare/Pclass) found for PDP.")
else:
    cols = st.columns(len(pdp_cols))
    for ax_col, feat in zip(cols, pdp_cols):
        fig_pdp, ax_pdp = plt.subplots(figsize=(4, 3))
        try:
            PartialDependenceDisplay.from_estimator(
                full_pipeline,
                X,
                features=[feat],
                ax=ax_pdp,
            )
            ax_pdp.set_title(f"PDP – {feat}")
            ax_col.pyplot(fig_pdp)
        except Exception as e:
            ax_col.write(f"Could not compute PDP for {feat}: {e}")

# --------------------------------------------------------
# SHAP feature importance (custom beeswarm)
# --------------------------------------------------------
st.subheader("🧠 SHAP Feature Importance (Custom Beeswarm)")
try:
    # Use TreeExplainer on underlying RandomForest
    rf = rf_model
    # Sample for speed
    sample_size = min(300, X_train.shape[0])
    X_sample = X_train.sample(sample_size, random_state=42)
    X_sample_proc = prep_fs.transform(X_sample)

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_sample_proc)

    # For binary classification, shap_values is a list of two arrays
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    # Custom beeswarm with matplotlib so it fits nicely in Streamlit
    fig_shap, ax_shap = plt.subplots(figsize=(7, 5))
    shap.summary_plot(
        sv,
        features=X_sample_proc,
        feature_names=feature_names,
        show=False,
        plot_type="dot",
        max_display=10,
    )
    st.pyplot(fig_shap)
except Exception as e:
    st.error(f"SHAP failed: {e}")

# --------------------------------------------------------
# Download cleaned dataset & simple HTML report
# --------------------------------------------------------
st.subheader("📥 Downloads")

cleaned_csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download cleaned dataset (CSV)",
    data=cleaned_csv,
    file_name="cleaned_dataset.csv",
    mime="text/csv",
)

# Simple HTML report
report_html = f"""
<html>
<head><title>Auto Data Toolkit Report</title></head>
<body>
<h1>Auto Data Toolkit Report</h1>
<h2>Dataset</h2>
<ul>
  <li>Original rows: {n_before}</li>
  <li>After duplicate removal & outliers: {len(df)}</li>
</ul>
<h2>Model</h2>
<ul>
  <li>RandomForestClassifier (n_estimators=300)</li>
  <li>Test size: {test_size}</li>
</ul>
<h2>Metrics</h2>
<ul>
  <li>Accuracy: {acc:.3f}</li>
  <li>Precision: {prec:.3f}</li>
  <li>Recall: {rec:.3f}</li>
  <li>F1-score: {f1:.3f}</li>
  <li>AUC: {auc:.3f}</li>
</ul>
<pre>
{report_text}
</pre>
</body>
</html>
"""
report_bytes = report_html.encode("utf-8")
st.download_button(
    "Download HTML model report",
    data=report_bytes,
    file_name="auto_toolkit_report.html",
    mime="text/html",
)

