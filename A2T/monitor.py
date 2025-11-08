#!/usr/bin/env python3
"""
monitor.py
Reads scored batches from datamart/gold/scored/, joins labels from
datamart/gold/label_store/ when available, and plots a line chart with:
 - ROC-AUC (stability) and
 - Positive Rate (operational/calibration proxy)
on the same figure (dual y-axes).

Outputs:
 - reports/metrics_trend.csv
 - reports/metrics_trend.png

Usage:
  python monitor.py
  python monitor.py --scored-dir datamart/gold/scored --labels-dir datamart/gold/label_store --prefer xgb
"""

import argparse, glob, re
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# -------------------- config --------------------
DEFAULT_SCORED_DIR = Path("datamart/gold/scored")
DEFAULT_LABELS_DIR = Path("datamart/gold/label_store")
REPORTS_DIR = Path("reports")

# -------------------- io helpers --------------------
def read_any(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.is_dir():
        return pd.read_parquet(p)
    if p.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(p)
    return pd.read_csv(p)

def list_scored(scored_dir: Path) -> list[Path]:
    files = sorted(
        list(scored_dir.glob("score_*.parquet")) + list(scored_dir.glob("score_*.csv")),
        key=lambda p: p.stat().st_mtime
    )
    return files

def list_label_files(labels_dir: Path) -> list[Path]:
    if not labels_dir.exists():
        return []
    return sorted(list(labels_dir.glob("*.parquet")) + list(labels_dir.glob("*.csv")))

def load_all_labels(labels_dir: Path) -> pd.DataFrame | None:
    files = list_label_files(labels_dir)
    if not files:
        return None
    dfs = []
    for f in files:
        df = read_any(f)
        # normalize schema best-effort
        if "snapshot_date" in df.columns:
            df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
        if "label" in df.columns:
            df["label"] = pd.to_numeric(df["label"], errors="coerce").astype("Int64")
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    return out

# -------------------- metrics --------------------
def batch_key_from_filename(path: Path) -> pd.Timestamp:
    """
    Extract a timestamp from 'score_YYYYMMDD_HHMMSS' if present,
    else use file mtime.
    """
    m = re.search(r"score_(\d{8})_(\d{6})", path.name)
    if m:
        dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        return pd.Timestamp(dt)
    return pd.Timestamp(path.stat().st_mtime, unit="s")

def choose_primary_model(cols: list[str], prefer: str | None) -> str:
    # prefer explicit request, else xgb, else lr
    if prefer and f"proba_{prefer}" in cols:
        return prefer
    return "xgb" if "proba_xgb" in cols else "lr"

def compute_auc_if_possible(df_scored: pd.DataFrame, df_labels: pd.DataFrame | None, id_cols: list[str]) -> dict[str, float | None]:
    """
    Try to compute AUC for available models by joining labels.
    Keys tried in order: snapshot_date + loan_id/customer_id; else loan_id; else customer_id.
    """
    if df_labels is None or "label" not in df_labels.columns:
        return {"lr": None, "xgb": None}

    # infer join keys
    keys = []
    if "snapshot_date" in df_scored.columns and "snapshot_date" in df_labels.columns:
        keys = ["snapshot_date"]
    join_on = None
    for candidate in [["loan_id"], ["customer_id"], ["loan_id", "customer_id"]]:
        if all(c in df_scored.columns for c in candidate) and all(c in df_labels.columns for c in candidate):
            join_on = (keys + candidate) if keys else candidate
            break
    if join_on is None:
        return {"lr": None, "xgb": None}

    left = df_scored.copy()
    if "snapshot_date" in left.columns:
        left["snapshot_date"] = pd.to_datetime(left["snapshot_date"], errors="coerce")

    right = df_labels[join_on + ["label"]].drop_duplicates()
    merged = left.merge(right, on=join_on, how="left")
    if merged["label"].isna().all():
        return {"lr": None, "xgb": None}

    out = {"lr": None, "xgb": None}
    for model in ["lr", "xgb"]:
        proba_col = f"proba_{model}"
        if proba_col in merged.columns and "label" in merged.columns:
            y_true = pd.to_numeric(merged["label"], errors="coerce")
            y_prob = pd.to_numeric(merged[proba_col], errors="coerce")
            valid = ~(y_true.isna() | y_prob.isna())
            if valid.sum() >= 10 and y_true.nunique() == 2:
                out[model] = float(roc_auc_score(y_true[valid], y_prob[valid]))
    return out

# -------------------- main flow --------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored-dir", default=str(DEFAULT_SCORED_DIR))
    ap.add_argument("--labels-dir", default=str(DEFAULT_LABELS_DIR))
    ap.add_argument("--prefer", choices=["xgb", "lr"], help="Primary model for positive-rate line if both present")
    ap.add_argument("--min_points", type=int, default=2, help="Require at least this many batches to plot")
    return ap.parse_args()

def main():
    args = parse_args()
    scored_dir = Path(args.scored_dir)
    labels_dir = Path(args.labels_dir)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    scored_files = list_scored(scored_dir)
    if len(scored_files) < args.min_points:
        raise SystemExit(f"Need at least {args.min_points} scored files in {scored_dir} to draw a meaningful line chart.")

    labels = load_all_labels(labels_dir) if labels_dir.exists() else None

    rows = []
    for f in scored_files:
        df = read_any(f)
        ts = batch_key_from_filename(f)
        # positive rates (if preds exist)
        pos = {f"pos_rate_{m}": float(df[f"pred_{m}"].mean()) for m in ["lr", "xgb"] if f"pred_{m}" in df.columns}
        meanp = {f"mean_proba_{m}": float(df[f"proba_{m}"].mean()) for m in ["lr", "xgb"] if f"proba_{m}" in df.columns}
        aucs = compute_auc_if_possible(df, labels, id_cols=["loan_id","customer_id","snapshot_date"])
        rows.append({"batch_ts": ts, **pos, **meanp, **{f"auc_{k}": v for k, v in aucs.items()}})

    trend = pd.DataFrame(rows).sort_values("batch_ts").reset_index(drop=True)
    # choose primary model for plotting second metric (positive rate)
    available_cols = trend.columns.tolist()
    primary = choose_primary_model(available_cols, args.prefer)

    # derive series to plot
    auc_col  = f"auc_{primary}"
    rate_col = f"pos_rate_{primary}"
    if auc_col not in trend or rate_col not in trend:
        # fallback gracefully: try other model or mean_proba
        other = "lr" if primary == "xgb" else "xgb"
        auc_col = f"auc_{primary}" if f"auc_{primary}" in trend else f"auc_{other}"
        rate_col = rate_col if rate_col in trend else (f"pos_rate_{other}" if f"pos_rate_{other}" in trend else f"mean_proba_{primary}")

    # Save CSV
    csv_path = REPORTS_DIR / "metrics_trend.csv"
    trend.to_csv(csv_path, index=False)

    # Require at least 2 non-null points for both lines
    if (trend[auc_col].notna().sum() < 2) or (trend[rate_col].notna().sum() < 2):
        print(f"Found {trend[auc_col].notna().sum()} AUC points and {trend[rate_col].notna().sum()} rate points; need ≥ 2 each for a line chart.")
        print(f"CSV written to {csv_path}. Add more batches/labels to enable plotting.")
        return

    # Plot (dual y-axes)
    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    ax2 = ax1.twinx()

    x = trend["batch_ts"]
    ax1.plot(x, trend[auc_col], marker="o", linewidth=2)
    ax2.plot(x, trend[rate_col], marker="s", linestyle="--", linewidth=2)

    ax1.set_xlabel("Batch Timestamp")
    ax1.set_ylabel("ROC-AUC", rotation=90)
    ax2.set_ylabel("Positive Rate", rotation=270, labelpad=15)

    title_model = primary.upper()
    ax1.set_title(f"Stability & Load over Time — {title_model}\n(AUC vs Positive Rate)")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()

    png_path = REPORTS_DIR / "metrics_trend.png"
    fig.savefig(png_path, dpi=160)
    print(f"✅ Trend written: {csv_path}")
    print(f"🖼️ Plot saved:   {png_path}")
    # Derive a cleaner month column for presentation
    trend["batch_month"] = pd.to_datetime(trend["batch_ts"]).dt.to_period("M").astype(str)
    # Save monitoring results as a GOLD table
    gold_dir = Path("datamart/gold/monitoring"); gold_dir.mkdir(parents=True, exist_ok=True)
    trend_path = gold_dir / "metrics_trend.parquet"
    trend.to_parquet(trend_path, index=False)
    print(f"🏆 Monitoring gold table written: {trend_path}")

if __name__ == "__main__":
    main()
