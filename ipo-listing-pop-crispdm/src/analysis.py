#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IPO Listing-Day Pop Prediction — CRISP-DM Pipeline
EDA -> Preprocessing -> Feature Engineering -> Modeling -> Evaluation
"""
import argparse, os, re, warnings
from pathlib import Path
from typing import Any, Tuple
import numpy as np, pandas as pd, matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, RocCurveDisplay
)

warnings.filterwarnings("ignore")
np.random.seed(42)
plt.rcParams["figure.figsize"] = (8,5)

# ---------- Utils ----------
num_regex = re.compile(r"-?\d+(?:\.\d+)?")
def to_snake(name: Any) -> str:
    return str(name).strip().replace("/", " ").replace("-", " ").replace("\n", " ").lower()

def parse_numeric_series(s: pd.Series) -> pd.Series:
    def parse_one(x):
        if pd.isna(x): return np.nan
        txt = str(x)
        neg = txt.strip().startswith("(") and txt.strip().endswith(")")
        m = num_regex.search(txt.replace(",", ""))
        if not m: return np.nan
        val = float(m.group(0))
        return -val if neg else val
    return s.apply(parse_one)

def iqr_clip(s: pd.Series, k: float = 1.5):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - k*iqr, q3 + k*iqr
    return s.clip(lower=lo, upper=hi), {"q1": float(q1), "q3": float(q3), "low": float(lo), "high": float(hi)}

def pick_best_sheet(xlsx_path: str):
    xls = pd.ExcelFile(xlsx_path)
    best_sheet, best_score, best_df = None, -1, None
    for s in xls.sheet_names:
        try:
            tmp = pd.read_excel(xlsx_path, sheet_name=s, engine="openpyxl")
            score = len(tmp) * max(1, tmp.shape[1])
            if score > best_score: best_sheet, best_score, best_df = s, score, tmp
        except Exception: pass
    assert best_df is not None, "No readable sheets found."
    return best_sheet, best_df

def chronological_split(dates: pd.Series, frac: float = 0.8):
    order = np.argsort(pd.to_datetime(dates, errors="coerce").values.astype("datetime64[ns]"))
    cut = int(frac * len(order))
    return order[:cut], order[cut:]

# ---------- EDA ----------
def run_eda(df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    quality = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "non_null": [df[c].notna().sum() for c in df.columns],
        "missing": [df[c].isna().sum() for c in df.columns]
    })
    quality["missing_pct"] = (quality["missing"]/len(df)*100).round(2)
    quality.sort_values("missing_pct", ascending=False).to_csv(outdir/"quality_audit.csv", index=False)

    key_num = [c for c in ["issue_size(crores)_num","offer_price_num","list_price_num","listing_gain_num",
                           "cmp(bse)_num","cmp(nse)_num","qib_num","hni_num","rii_num","total_num"] if c in df.columns]
    for c in key_num[:6]:
        plt.figure()
        df[c].dropna().hist(bins=30)
        plt.title(f"Distribution of {c}"); plt.xlabel(c); plt.ylabel("Count")
        plt.tight_layout(); plt.savefig(outdir/f"hist_{c}.png"); plt.close()

    pairs = [("offer_price_num","list_price_num"),("issue_size(crores)_num","listing_gain_num"),("qib_num","listing_gain_num")]
    for x,y in pairs:
        if x in df.columns and y in df.columns:
            plt.figure()
            plt.scatter(df[x], df[y], s=12, alpha=0.7)
            plt.xlabel(x); plt.ylabel(y); plt.title(f"{y} vs {x}")
            plt.tight_layout(); plt.savefig(outdir/f"scatter_{y}_vs_{x}.png"); plt.close()

    dcols = [c for c in df.columns if c.endswith("_parsed") and "date" in c]
    if dcols and "listing_gain_num" in df.columns:
        dcol = dcols[0]
        tmp = df[[dcol,"listing_gain_num"]].dropna()
        if len(tmp):
            tmp["year"] = tmp[dcol].dt.year
            yr = tmp.groupby("year")["listing_gain_num"].mean().reset_index()
            yr.to_csv(outdir/"avg_listing_gain_by_year.csv", index=False)
            plt.figure(); plt.plot(yr["year"], yr["listing_gain_num"])
            plt.title("Average listing gain by year"); plt.xlabel("year"); plt.ylabel("avg listing gain")
            plt.tight_layout(); plt.savefig(outdir/"trend_avg_listing_gain_by_year.png"); plt.close()

# ---------- Prep + Feature Engineering ----------
def prepare_and_engineer(df_raw: pd.DataFrame):
    df = df_raw.copy()
    df.columns = [to_snake(c).replace(" ", "_") for c in df.columns]
    for c in df.columns:
        if "date" in c and not c.endswith("_parsed"):
            d = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
            if d.notna().mean() > 0.6: df[c+"_parsed"] = d
    hints = ["price","gain","size","cmp","qib","hni","rii","total"]
    for c in df.columns:
        if any(h in c for h in hints) or pd.api.types.is_numeric_dtype(df[c]):
            df[c+"_num"] = parse_numeric_series(df[c])

    drop_cols = [c for c in df.columns if c.startswith("unnamed")]
    for leak in ["current_gains","current_gains_num"]:
        if leak in df.columns: drop_cols.append(leak)
    work = df.drop(columns=list(set(drop_cols)), errors="ignore").copy()

    num_candidates = [c for c in ["issue_size(crores)_num","offer_price_num","list_price_num","listing_gain_num",
                                  "cmp(bse)_num","cmp(nse)_num","qib_num","hni_num","rii_num","total_num"] if c in work.columns]
    for c in num_candidates: work[c] = work[c].fillna(work[c].median())

    wins = [c for c in ["issue_size(crores)_num","offer_price_num","list_price_num","cmp(bse)_num","cmp(nse)_num","listing_gain_num"] if c in work.columns]
    for c in wins: work[c], _ = iqr_clip(work[c])

    if num_candidates:
        iso = IsolationForest(random_state=42, contamination=0.03)
        work["iso_outlier"] = (iso.fit_predict(work[num_candidates])==-1).astype(int)

    dcols = [c for c in work.columns if c.endswith("_parsed") and "date" in c]
    if dcols:
        dcol = dcols[0]
        work["year"] = work[dcol].dt.year
        work["quarter"] = work[dcol].dt.quarter
        work["month"] = work[dcol].dt.month
        work["post_2020"] = (work["year"]>=2020).astype(int)
    else:
        work["year"]=np.nan; work["quarter"]=np.nan; work["month"]=np.nan; work["post_2020"]=np.nan

    for c in ["qib_num","hni_num","rii_num"]:
        if c not in work.columns: work[c] = np.nan
    work["total_calc_num"] = work[["qib_num","hni_num","rii_num"]].sum(axis=1, skipna=True)
    work["total_filled_num"] = work["total_num"].fillna(work["total_calc_num"]) if "total_num" in work.columns else work["total_calc_num"]
    eps = 1e-6
    work["qib_share"] = work["qib_num"]/(work["total_filled_num"]+eps)
    work["hni_share"] = work["hni_num"]/(work["total_filled_num"]+eps)
    work["rii_share"] = work["rii_num"]/(work["total_filled_num"]+eps)
    thr = work["total_filled_num"].quantile(0.75)
    work["hot_demand"] = (work["total_filled_num"]>=thr).astype(int)

    if "issue_size(crores)_num" in work.columns:
        work["size_log1p"] = np.log1p(work["issue_size(crores)_num"].fillna(work["issue_size(crores)_num"].median()))
        try:
            work["size_bin"] = pd.qcut(work["issue_size(crores)_num"], q=3, labels=["small","mid","large"])
        except Exception:
            work["size_bin"] = pd.cut(work["issue_size(crores)_num"], bins=3, labels=["small","mid","large"])
    else:
        work["size_log1p"] = np.nan; work["size_bin"] = "small"

    work["offer_log1p"] = np.log1p(work["offer_price_num"]) if "offer_price_num" in work.columns else np.nan
    work["size_x_total"] = work["size_log1p"]*np.log1p(work["total_filled_num"]+1)
    work["qib_x_offer"] = work["offer_log1p"]*np.log1p(work["qib_num"]+1)
    work["rii_to_qib_ratio"] = work["rii_num"]/(work["qib_num"]+eps)

    X_cat = pd.get_dummies(work[["size_bin","quarter"]], drop_first=True) if all(c in work.columns for c in ["size_bin","quarter"]) else pd.DataFrame(index=work.index)
    num_cols = [c for c in ["year","month","post_2020","qib_share","hni_share","rii_share","total_filled_num",
                            "size_log1p","size_x_total","qib_x_offer","rii_to_qib_ratio","offer_log1p",
                            "qib_num","hni_num","rii_num"] if c in work.columns]
    num_filled = work[num_cols].copy()
    for c in num_cols: num_filled[c] = num_filled[c].fillna(num_filled[c].median())
    scaler = RobustScaler()
    X_num = pd.DataFrame(scaler.fit_transform(num_filled), columns=[f"{c}__rsc" for c in num_cols], index=work.index)
    X = pd.concat([X_num, X_cat], axis=1)

    if "listing_gain_num" not in work.columns:
        raise ValueError("Expected 'listing_gain' column (parsed to listing_gain_num).")
    y = (work["listing_gain_num"]>0).astype(int)
    return X, y, work

# ---------- Modeling + Evaluation ----------
def train_and_evaluate(X: pd.DataFrame, y: pd.Series, work: pd.DataFrame, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    dcols = [c for c in work.columns if c.endswith("_parsed") and "date" in c]
    assert dcols, "No parsed date column found for chronological split."
    dates = work[dcols[0]]
    order = np.argsort(dates.values.astype("datetime64[ns]"))
    cut = int(0.8*len(order))
    train_idx, test_idx = order[:cut], order[cut:]
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    if y_test.nunique() < 2:
        cut = int(0.75*len(order))
        train_idx, test_idx = order[:cut], order[cut:]
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_split=4, min_samples_leaf=2,
            random_state=42, n_jobs=-1, class_weight="balanced_subsample"
        ),
    }

    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=42, eval_metric="logloss", n_jobs=-1
        )
    except Exception:
        models["GradientBoosting"] = GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.1, max_depth=3, random_state=42
        )

    models["MLP"] = MLPClassifier(
        hidden_layer_sizes=(32,8), activation="relu", solver="adam",
        alpha=1e-3, max_iter=300, early_stopping=True, random_state=42
    )

    rows, probs, fitted = [], {}, {}
    def proba_of(model, X_):
        return model.predict_proba(X_)[:,1] if hasattr(model, "predict_proba") else model.decision_function(X_)

    for name, clf in models.items():
        clf.fit(X_train, y_train)
        p_tr, p_te = proba_of(clf, X_train), proba_of(clf, X_test)
        yhat_tr, yhat_te = (p_tr >= 0.5).astype(int), (p_te >= 0.5).astype(int)
        rows.append({
            "Model": name,
            "train_Accuracy": accuracy_score(y_train, yhat_tr),
            "train_Precision": precision_score(y_train, yhat_tr, zero_division=0),
            "train_Recall": recall_score(y_train, yhat_tr, zero_division=0),
            "train_F1": f1_score(y_train, yhat_tr, zero_division=0),
            "train_ROC_AUC": roc_auc_score(y_train, p_tr),
            "test_Accuracy": accuracy_score(y_test, yhat_te),
            "test_Precision": precision_score(y_test, yhat_te, zero_division=0),
            "test_Recall": recall_score(y_test, yhat_te, zero_division=0),
            "test_F1": f1_score(y_test, yhat_te, zero_division=0),
            "test_ROC_AUC": roc_auc_score(y_test, p_te),
        })
        probs[name] = p_te
        fitted[name] = clf

        cm = confusion_matrix(y_test, yhat_te)
        plt.figure()
        plt.imshow(cm, interpolation="nearest")
        plt.title(f"Confusion Matrix — {name} (Test)")
        plt.xticks([0,1], ["Pred 0","Pred 1"]); plt.yticks([0,1], ["True 0","True 1"])
        for (i,j), z in np.ndenumerate(cm):
            plt.text(j, i, str(z), ha='center', va='center')
        plt.tight_layout(); plt.savefig(outdir/f"cm_{name}.png"); plt.close()

    import pandas as pd
    metrics_df = pd.DataFrame(rows).assign(
        gen_gap_AUC=lambda d: d["train_ROC_AUC"] - d["test_ROC_AUC"]
    ).sort_values("test_ROC_AUC", ascending=False)
    metrics_df.to_csv(outdir / "model_metrics.csv", index=False)

    plt.figure()
    for name, p in probs.items():
        RocCurveDisplay.from_predictions(y_test, p, name=name)
    plt.title("ROC Curves — Chronological Test Split")
    plt.tight_layout()
    plt.savefig(outdir / "roc_curves.png")
    plt.close()

    best_name = metrics_df.iloc[0]["Model"]
    print("Best by test ROC-AUC:", best_name)

    import joblib
    (outdir.parent / "artifacts").mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[best_name], outdir.parent / "artifacts" / f"model_{best_name}.joblib")
    (outdir.parent / "artifacts" / "feature_columns.csv").write_text("\\n".join(X.columns), encoding="utf-8")

    return metrics_df

def main():
    parser = argparse.ArgumentParser(description="IPO Listing-Day Pop Prediction — CRISP-DM Pipeline")
    parser.add_argument("--data", type=str, default="data/Initial Public Offering.xlsx", help="Path to Excel dataset")
    parser.add_argument("--reports_dir", type=str, default="reports", help="Directory for output reports/plots")
    args = parser.parse_args()

    xlsx_path = Path(args.data)
    assert xlsx_path.exists(), f"Excel file not found at {xlsx_path}"
    sheet, df_raw = pick_best_sheet(str(xlsx_path))
    print(f"[LOAD] Selected sheet: {sheet} shape={df_raw.shape}")

    df = df_raw.copy()
    df.columns = [to_snake(c).replace(" ", "_") for c in df.columns]
    for c in df.columns:
        if "date" in c and not c.endswith("_parsed"):
            d = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
            if d.notna().mean() > 0.6:
                df[c + "_parsed"] = d
    hints = ["price","gain","size","cmp","qib","hni","rii","total"]
    for c in df.columns:
        if any(h in c for h in hints) or pd.api.types.is_numeric_dtype(df[c]):
            df[c + "_num"] = parse_numeric_series(df[c])

    run_eda(df, Path(args.reports_dir))
    X, y, work = prepare_and_engineer(df)
    metrics = train_and_evaluate(X, y, work, Path(args.reports_dir))
    print("\\n[RESULTS]\\n", metrics.to_string(index=False))

if __name__ == "__main__":
    main()
