"""
project_pipeline.core

Small reusable pipeline module for the CMPE255 final project.
The Streamlit app builds the pipeline inline, but this module shows
how you might reuse the same idea from Python code or notebooks.
"""

from dataclasses import dataclass
from typing import Dict, Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


@dataclass
class ExperimentResult:
    metrics: Dict[str, float]
    model: Any


def build_pipeline(numeric_features, categorical_features) -> Pipeline:
    """Create a RandomForest pipeline with basic preprocessing."""
    num_tf = Pipeline([("scaler", StandardScaler())])
    cat_tf = Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore"))])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_tf, numeric_features),
            ("cat", cat_tf, categorical_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline([("pre", pre), ("model", model)])
    return pipe


def run_full_experiment(df: pd.DataFrame, target_col: str) -> ExperimentResult:
    """Run a simple train/test split experiment on a dataframe."""
    assert target_col in df.columns, f"target '{target_col}' not found"

    df = df.dropna(subset=[target_col]).copy()
    y = df[target_col]
    X = df.drop(columns=[target_col])

    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    pipe = build_pipeline(num_cols, cat_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, average="weighted")),
    }

    return ExperimentResult(metrics=metrics, model=pipe)
