#!/usr/bin/env python3
from __future__ import annotations
"""
infer.py
Score new data using one or both model bundles.
Auto-loads tuned thresholds from metrics CSVs produced by train.py.
Writes scored output into datamart/gold/scored/.

Usage:
  python infer.py \
    --input datamart/gold/model_dataset/gold_model_dataset_2024_12_01.parquet \
    --id-cols customer_id loan_id
"""

import argparse, re, glob
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

# --------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------
MODELS_DIR = Path("models")
DEFAULT_OUT_DIR = Path("datamart/gold/scored")

# --------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------
def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.is_dir():
        return pd.read_parquet(p)
    if p.suffix.lower() in [".parquet", ".pq"]:
        return pd.read_parquet(p)
    return pd.read_csv(p)

def write_table(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in [".parquet", ".pq"]:
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)

def ensure_columns(df: pd.DataFrame, expected_cols: list[str]) -> pd.DataFrame:
    missing = [c for c in expected_cols if c not in df.columns]
    for c in missing:
        df[c] = np.nan
    return df[expected_cols]

# --------------------------------------------------------------------
# LOAD MODELS
# --------------------------------------------------------------------
def load_latest_lr():
    files = sorted(MODELS_DIR.glob("logreg_pipeline_*.joblib"))
    if files:
        return joblib.load(files[-1]), files[-1]
    fb = MODELS_DIR / "logreg_pipeline.joblib"
    return (joblib.load(fb), fb) if fb.exists() else (None, None)

def load_latest_xgb():
    files = sorted(MODELS_DIR.glob("xgb_bundle_*.joblib"))
    if files:
        return joblib.load(files[-1]), files[-1]
    fb = MODELS_DIR / "xgb_bundle.joblib"
    return (joblib.load(fb), fb) if fb.exists() else (None, None)

# --------------------------------------------------------------------
# LOAD TUNED THRESHOLDS FROM METRICS CSV
# --------------------------------------------------------------------
def load_tuned_threshold(model_name: str) -> float | None:
    csv_path = MODELS_DIR / f"{model_name}_metrics_all_splits.csv"
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
        # Prefer OOT_tuned or Valid_tuned row
        tuned = df[df["split"].str.contains("tuned", case=False, na=False)]
        if tuned.empty:
            return None
        thr = tuned["threshold"].iloc[0]
        print(f"Loaded tuned threshold for {model_name.upper()} = {thr:.3f} (from {csv_path.name})")
        return float(thr)
    except Exception as e:
        print(f"⚠️  Could not read tuned threshold from {csv_path}: {e}")
        return None

# --------------------------------------------------------------------
# SCORING FUNCTIONS
# --------------------------------------------------------------------
def score_lr(bundle, df_feat: pd.DataFrame, thr: float | None):
    pipe = bundle["pipeline"]
    feats = bundle.get("feature_cols") or (bundle.get("num_cols", []) + bundle.get("cat_cols", []))
    X = ensure_columns(df_feat.copy(), feats)
    proba = pipe.predict_proba(X)[:, 1]
    use_thr = float(thr if thr is not None else bundle.get("tuned_threshold", 0.5))
    pred = (proba >= use_thr).astype(int)
    return proba, pred, use_thr

def score_xgb(bundle, df_feat: pd.DataFrame, thr: float | None):
    prep = bundle["preprocess"]
    booster = bundle["booster"]
    feats = bundle.get("feature_cols") or (bundle.get("num_cols", []) + bundle.get("cat_cols", []))
    X = ensure_columns(df_feat.copy(), feats)
    d = xgb.DMatrix(prep.transform(X))
    proba = booster.predict(d)
    use_thr = float(thr if thr is not None else bundle.get("tuned_threshold", 0.5))
    pred = (proba >= use_thr).astype(int)
    return proba, pred, use_thr

# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input file/directory to score")
    p.add_argument("--id-cols", nargs="*", default=[], help="ID columns to keep in output")
    p.add_argument("--bundle-lr", help="Optional explicit Logistic Regression bundle path")
    p.add_argument("--bundle-xgb", help="Optional explicit XGBoost bundle path")
    p.add_argument("--thr-lr", type=float, help="Override threshold for LR")
    p.add_argument("--thr-xgb", type=float, help="Override threshold for XGB")
    p.add_argument("--outdir", default=str(DEFAULT_OUT_DIR), help="Where to write scored outputs")
    return p.parse_args()

def main():
    args = parse_args()
    raw = read_table(args.input)
    ids = raw[args.id_cols].copy() if args.id_cols else pd.DataFrame()
    features_df = raw.copy()

    outputs, used = [], []

    # ---------------- Logistic Regression ----------------
    lr_bundle, lr_path = (joblib.load(args.bundle_lr), Path(args.bundle_lr)) if args.bundle_lr else load_latest_lr()
    if lr_bundle is not None:
        thr_lr = args.thr_lr or load_tuned_threshold("logreg") or lr_bundle.get("tuned_threshold", 0.5)
        proba, pred, thr = score_lr(lr_bundle, features_df, thr_lr)
        out = pd.concat([ids.reset_index(drop=True), pd.DataFrame({"proba_lr": proba, "pred_lr": pred})], axis=1)
        outputs.append(out)
        used.append(f"LR ({lr_path.name if lr_path else 'unspecified'}) thr={thr:.3f}")

    # ---------------- XGBoost ----------------
    xgb_bundle, xgb_path = (joblib.load(args.bundle_xgb), Path(args.bundle_xgb)) if args.bundle_xgb else load_latest_xgb()
    if xgb_bundle is not None:
        thr_xgb = args.thr_xgb or load_tuned_threshold("xgb") or xgb_bundle.get("tuned_threshold", 0.5)
        proba, pred, thr = score_xgb(xgb_bundle, features_df, thr_xgb)
        out = pd.concat([ids.reset_index(drop=True), pd.DataFrame({"proba_xgb": proba, "pred_xgb": pred})], axis=1)
        outputs.append(out)
        used.append(f"XGB ({xgb_path.name if xgb_path else 'unspecified'}) thr={thr:.3f}")

    if not outputs:
        raise SystemExit("No model bundles found. Check models/ or provide --bundle-* manually.")

    # Merge outputs
    final = outputs[0]
    for extra in outputs[1:]:
        final = pd.concat(
            [final, extra.drop(columns=[c for c in args.id_cols if c in extra.columns], errors="ignore")],
            axis=1
        )

    # ---------------- Write output ----------------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir)
    out_path = outdir / f"score_{ts}.parquet"
    write_table(final, out_path)

    print(f"✅ Scored {len(final)} rows → {out_path}")
    for u in used:
        print("   •", u)
    for col in [c for c in final.columns if c.startswith("pred_")]:
        print(f"   {col}: positive rate = {float(final[col].mean()):.4f}")

if __name__ == "__main__":
    main()
