#!/usr/bin/env python3
from __future__ import annotations
"""
train.py
Read gold model_dataset snapshots, make time-based splits, train Logistic Regression
(sklearn pipeline) and XGBoost (DMatrix early stopping + refit), evaluate across
splits, tune threshold on VALID (max F1), and save versioned artifacts + metrics.

Usage (default dates baked in, can override via CLI):
  python train_both.py

Optional args:
  --models both|logreg|xgb
  --outdir models
  --train-start 2023-07-01 --train-end 2024-04-30
  --valid-start 2024-05-01 --valid-end 2024-06-30
  --test-start  2024-07-01 --test-end  2024-10-31
  --oot-start   2024-11-01 --oot-end   2024-12-31
"""

import argparse, json, time, re
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, precision_recall_curve
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

CANDIDATES = [Path("gold/model_dataset"), Path("datamart/gold/model_dataset")]

def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # For older sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def find_snapshot_paths() -> list[str]:
    for base in CANDIDATES:
        pat = str(base / "gold_model_dataset_*.parquet")
        found = sorted(glob.glob(pat))
        if found:
            print(f"Found {len(found)} snapshots under: {base}")
            return found
    raise FileNotFoundError(
        "Couldn’t find any gold_model_dataset_*.parquet under "
        "'gold/model_dataset' or 'datamart/gold/model_dataset'."
    )

def read_all(paths: list[str]) -> pd.DataFrame:
    dfs = [pd.read_parquet(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    df["label"] = df["label"].astype(int)
    return df

def split_by_dates(df: pd.DataFrame, args):
    s = lambda d: pd.Timestamp(d)
    train = df[(df["snapshot_date"] >= s(args.train_start)) & (df["snapshot_date"] <= s(args.train_end))]
    valid = df[(df["snapshot_date"] >= s(args.valid_start)) & (df["snapshot_date"] <= s(args.valid_end))]
    test  = df[(df["snapshot_date"] >= s(args.test_start))  & (df["snapshot_date"] <= s(args.test_end))]
    oot   = df[(df["snapshot_date"] >= s(args.oot_start))   & (df["snapshot_date"] <= s(args.oot_end))]
    return train, valid, test, oot

def eval_split_lr(model, X, y, split_name, thr=0.5):
    proba = model.predict_proba(X)[:, 1]
    pred  = (proba >= thr).astype(int)
    roc = roc_auc_score(y, proba)
    pr  = average_precision_score(y, proba)
    rpt = classification_report(y, pred, digits=3, output_dict=True)
    cm  = confusion_matrix(y, pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n=== Logistic Regression — {split_name} (thr={thr:.3f}) ===")
    print(f"ROC-AUC: {roc:.3f} | PR-AUC: {pr:.3f}")
    print(f"Precision (1): {rpt['1']['precision']:.3f} | Recall (1): {rpt['1']['recall']:.3f} | F1 (1): {rpt['1']['f1-score']:.3f}")
    print(f"TN={tn:>4}  FP={fp:>4}\nFN={fn:>4}  TP={tp:>4}")
    return {
        "model":"logreg","split":split_name,"threshold":thr,
        "roc_auc":float(roc),"pr_auc":float(pr),
        "precision_1":float(rpt["1"]["precision"]),
        "recall_1":float(rpt["1"]["recall"]),
        "f1_1":float(rpt["1"]["f1-score"]),
        "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)
    }

def predict_proba_xgb(prep, booster, X):
    Xt = prep.transform(X)
    d = xgb.DMatrix(Xt)
    return booster.predict(d)

def eval_split_xgb(prep, booster, X, y, split_name, thr=0.5):
    proba = predict_proba_xgb(prep, booster, X)
    pred  = (proba >= thr).astype(int)
    roc = roc_auc_score(y, proba)
    pr  = average_precision_score(y, proba)
    rpt = classification_report(y, pred, digits=3, output_dict=True)
    cm  = confusion_matrix(y, pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n=== XGBoost — {split_name} (thr={thr:.3f}) ===")
    print(f"ROC-AUC: {roc:.3f} | PR-AUC: {pr:.3f}")
    print(f"Precision (1): {rpt['1']['precision']:.3f} | Recall (1): {rpt['1']['recall']:.3f} | F1 (1): {rpt['1']['f1-score']:.3f}")
    print(f"TN={tn:>4}  FP={fp:>4}\nFN={fn:>4}  TP={tp:>4}")
    return {
        "model":"xgb","split":split_name,"threshold":thr,
        "roc_auc":float(roc),"pr_auc":float(pr),
        "precision_1":float(rpt["1"]["precision"]),
        "recall_1":float(rpt["1"]["recall"]),
        "f1_1":float(rpt["1"]["f1-score"]),
        "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)
    }

def tune_threshold(y_valid, proba_valid):
    prec_v, rec_v, thr_v = precision_recall_curve(y_valid, proba_valid)
    f1_v = 2 * (prec_v * rec_v) / (prec_v + rec_v + 1e-9)
    best_idx = int(np.nanargmax(f1_v))
    best_thr = float(thr_v[best_idx])
    print(f"\n>>> Best threshold from VALID (max F1): {best_thr:.3f} "
          f"| Precision={prec_v[best_idx]:.3f}, Recall={rec_v[best_idx]:.3f}, F1={f1_v[best_idx]:.3f}")
    return best_thr

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", choices=["both","logreg","xgb"], default="both")
    p.add_argument("--outdir", default="models")
    # Date ranges (defaults follow your notebook)
    p.add_argument("--train-start", default="2023-07-01")
    p.add_argument("--train-end",   default="2024-04-30")
    p.add_argument("--valid-start", default="2024-05-01")
    p.add_argument("--valid-end",   default="2024-06-30")
    p.add_argument("--test-start",  default="2024-07-01")
    p.add_argument("--test-end",    default="2024-10-31")
    p.add_argument("--oot-start",   default="2024-11-01")
    p.add_argument("--oot-end",     default="2024-12-31")
    return p.parse_args()

def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # 1) Load data
    paths = find_snapshot_paths()
    df = read_all(paths)
    print("Rows:", len(df), "Cols:", df.shape[1])

    # 2) Time splits
    train_df, valid_df, test_df, oot_df = split_by_dates(df, args)
    print(f"Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}, OOT: {len(oot_df)}")

    # 3) Drop ID-like columns
    drop_cols = ["customer_id", "loan_id", "label_def"]
    train_df = train_df.drop(columns=drop_cols, errors="ignore")
    valid_df = valid_df.drop(columns=drop_cols, errors="ignore")
    test_df  = test_df.drop(columns=drop_cols,  errors="ignore")
    oot_df   = oot_df.drop(columns=drop_cols,   errors="ignore")

    # 4) Feature matrix selection (match notebook)
    feature_cols = [c for c in df.columns if c not in ["label","snapshot_date","customer_id","loan_id","label_def"]]
    X_train, y_train = train_df[feature_cols], train_df["label"].astype(int)
    X_valid, y_valid = valid_df[feature_cols], valid_df["label"].astype(int)
    X_test,  y_test  = test_df[feature_cols],  test_df["label"].astype(int)
    X_oot,   y_oot   = oot_df[feature_cols],   oot_df["label"].astype(int)

    # Column types (derived from TRAIN)
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object","category","bool"]).columns.tolist()

    numeric_tf = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_tf = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", make_ohe())])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = []

    # ===== Logistic Regression =====
    if args.models in ("both","logreg"):
        preprocess_lr = ColumnTransformer([("num", numeric_tf, num_cols),
                                           ("cat", categorical_tf, cat_cols)])
        pipe_lr = Pipeline([("prep", preprocess_lr),
                            ("clf", LogisticRegression(max_iter=1000, solver="lbfgs"))])

        # Fit on Train + Valid
        X_trv = pd.concat([X_train, X_valid], axis=0)
        y_trv = pd.concat([y_train, y_valid], axis=0)
        pipe_lr.fit(X_trv, y_trv)

        # Default-threshold eval
        lr_metrics = []
        lr_metrics.append(eval_split_lr(pipe_lr, X_train, y_train, "Train", thr=0.5))
        lr_metrics.append(eval_split_lr(pipe_lr, X_valid, y_valid, "Valid", thr=0.5))
        lr_metrics.append(eval_split_lr(pipe_lr, X_test,  y_test,  "Test",  thr=0.5))
        lr_metrics.append(eval_split_lr(pipe_lr, X_oot,   y_oot,   "OOT",   thr=0.5))

        # Tune threshold on VALID (max F1)
        proba_valid = pipe_lr.predict_proba(X_valid)[:, 1]
        lr_thr = tune_threshold(y_valid, proba_valid)

        # Re-eval with tuned threshold
        lr_metrics.append(eval_split_lr(pipe_lr, X_train, y_train, "Train_tuned", thr=lr_thr))
        lr_metrics.append(eval_split_lr(pipe_lr, X_valid, y_valid, "Valid_tuned", thr=lr_thr))
        lr_metrics.append(eval_split_lr(pipe_lr, X_test,  y_test,  "Test_tuned",  thr=lr_thr))
        lr_metrics.append(eval_split_lr(pipe_lr, X_oot,   y_oot,   "OOT_tuned",   thr=lr_thr))

        # Save artifact (pipeline includes preprocess)
        lr_path = outdir / f"logreg_pipeline_{ts}.joblib"
        lr_bundle = {
            "pipeline": pipe_lr,
            "num_cols": num_cols, "cat_cols": cat_cols, "feature_cols": feature_cols,
            "created_at": ts, "model_type":"logreg",
            "tuned_threshold": lr_thr
        }
        joblib.dump(lr_bundle, lr_path)
        print("✅ Saved:", lr_path)

        # Save metrics
        lr_df = pd.DataFrame(lr_metrics)
        lr_df.to_csv(outdir / "logreg_metrics_all_splits.csv", index=False)
        print("📄 Saved metrics:", outdir / "logreg_metrics_all_splits.csv")

        summary.append({"model":"logreg", "bundle": str(lr_path),
                        "roc_auc_valid": float(lr_metrics[1]["roc_auc"]),
                        "pr_auc_valid":  float(lr_metrics[1]["pr_auc"]),
                        "tuned_threshold": lr_thr})

    # ===== XGBoost =====
    if args.models in ("both","xgb"):
        # Stage 1: early stopping on VALID
        pre_stage1 = ColumnTransformer([("num", numeric_tf, num_cols),
                                        ("cat", categorical_tf, cat_cols)])
        Xtr = pre_stage1.fit_transform(X_train, y_train)
        Xva = pre_stage1.transform(X_valid)

        dtr = xgb.DMatrix(Xtr, label=y_train.to_numpy())
        dva = xgb.DMatrix(Xva, label=y_valid.to_numpy())

        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "eta": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "seed": 42,
        }
        bst_es = xgb.train(params, dtr, num_boost_round=2000, evals=[(dva, "valid")],
                           early_stopping_rounds=50, verbose_eval=False)
        best_iter = int(bst_es.best_iteration + 1)
        print("Best iteration (early stop):", best_iter)

        # Stage 2: refit on Train+Valid
        pre_final = ColumnTransformer([("num", numeric_tf, num_cols),
                                       ("cat", categorical_tf, cat_cols)])
        X_trv = pd.concat([X_train, X_valid], axis=0)
        y_trv = pd.concat([y_train, y_valid], axis=0)
        Xtrv = pre_final.fit_transform(X_trv, y_trv)
        dtrv = xgb.DMatrix(Xtrv, label=y_trv.to_numpy())
        bst_final = xgb.train(params, dtrv, num_boost_round=best_iter, verbose_eval=False)

        # Default-threshold eval
        xgb_metrics = []
        xgb_metrics.append(eval_split_xgb(pre_final, bst_final, X_train, y_train, "Train", thr=0.5))
        xgb_metrics.append(eval_split_xgb(pre_final, bst_final, X_valid, y_valid, "Valid", thr=0.5))
        xgb_metrics.append(eval_split_xgb(pre_final, bst_final, X_test,  y_test,  "Test",  thr=0.5))
        xgb_metrics.append(eval_split_xgb(pre_final, bst_final, X_oot,   y_oot,   "OOT",   thr=0.5))

        # Tune threshold on VALID
        proba_valid = predict_proba_xgb(pre_final, bst_final, X_valid)
        xgb_thr = tune_threshold(y_valid, proba_valid)

        # Re-eval tuned
        xgb_metrics.append(eval_split_xgb(pre_final, bst_final, X_train, y_train, "Train_tuned", thr=xgb_thr))
        xgb_metrics.append(eval_split_xgb(pre_final, bst_final, X_valid, y_valid, "Valid_tuned", thr=xgb_thr))
        xgb_metrics.append(eval_split_xgb(pre_final, bst_final, X_test,  y_test,  "Test_tuned",  thr=xgb_thr))
        xgb_metrics.append(eval_split_xgb(pre_final, bst_final, X_oot,   y_oot,   "OOT_tuned",   thr=xgb_thr))

        # Save artifact
        xgb_path = outdir / f"xgb_bundle_{ts}.joblib"
        xgb_bundle = {
            "preprocess": pre_final,
            "booster": bst_final,
            "num_cols": num_cols, "cat_cols": cat_cols, "feature_cols": feature_cols,
            "best_iter": best_iter, "created_at": ts, "model_type":"xgb",
            "tuned_threshold": xgb_thr
        }
        joblib.dump(xgb_bundle, xgb_path)
        print("✅ Saved:", xgb_path)

        # Save metrics
        xgb_df = pd.DataFrame(xgb_metrics)
        xgb_df.to_csv(outdir / "xgb_metrics_all_splits.csv", index=False)
        print("📄 Saved metrics:", outdir / "xgb_metrics_all_splits.csv")

        summary.append({"model":"xgb", "bundle": str(xgb_path),
                        "roc_auc_valid": float(xgb_metrics[1]["roc_auc"]),
                        "pr_auc_valid":  float(xgb_metrics[1]["pr_auc"]),
                        "tuned_threshold": xgb_thr})

    # Comparison
    cmp_path = outdir / f"compare_{ts}.json"
    with open(cmp_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\n📊 Comparison saved to:", cmp_path)
    for r in summary:
        print(f" - {r['model']:<6} valid AUC={r['roc_auc_valid']:.4f} PR-AUC={r['pr_auc_valid']:.4f} thr*={r['tuned_threshold']:.3f}")

if __name__ == "__main__":
    import numpy as np  # ensure available for tune_threshold
    main()
