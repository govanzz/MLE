# utils/data_processing_bronze_all.py
# Combined Bronze ingest for: LMS (labels source) + Feature datasets
# - Mirrors your Lab 2 Bronze style (CSV partitions, minimal transforms)
# - Adds lineage cols for features: ingest_dt, snap_ym, source_file
# - Tries to filter by snapshot month if time columns exist

import os
from datetime import datetime
import pyspark.sql.functions as F
from pyspark.sql.functions import col

# -----------------------------
# Helpers
# -----------------------------
def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _ym(date_str: str) -> str:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return f"{dt.year:04d}-{dt.month:02d}"

def _find_file(raw_dir: str, candidates: list[str]) -> str | None:
    for name in candidates:
        p = os.path.join(raw_dir, name)
        if os.path.exists(p):
            return p
    return None

def _partition_csv_path(base_dir: str, prefix: str, snapshot_date_str: str) -> str:
    fname = f"{prefix}_{snapshot_date_str.replace('-','_')}.csv"
    return os.path.join(base_dir, fname)

# -----------------------------
# 1) LMS Bronze (your original logic)
# -----------------------------
def process_bronze_lms(snapshot_date_str: str, bronze_lms_directory: str, spark):
    """
    Reads data/lms_loan_daily.csv, filters by snapshot_date == snapshot_date_str,
    writes CSV partition to bronze_lms_directory (your Lab 2 behavior).
    """
    _ensure_dir(bronze_lms_directory)
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    csv_file_path = "data/lms_loan_daily.csv"
    df = (
        spark.read.csv(csv_file_path, header=True, inferSchema=True)
        .filter(col("snapshot_date") == snapshot_date)
    )
    print(snapshot_date_str + " row count:", df.count())

    filepath = _partition_csv_path(bronze_lms_directory, "bronze_loan_daily", snapshot_date_str)
    df.toPandas().to_csv(filepath, index=False)
    print("saved to:", filepath)

# -----------------------------
# 2) Features Bronze
# -----------------------------
def _filter_to_snapshot_if_possible(df, snapshot_date_str: str, dataset_name: str):
    """
    Try to restrict to the snapshot month if a suitable time column exists.
    Priority per dataset:
      - clickstream: event_ts or event_date
      - financials:  txn_ts or txn_date
      - attributes:  signup_date
      - generic:     snapshot_date
    If none exist, returns df unchanged (full file).
    """
    snapshot_dt = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    first_of_month = datetime(snapshot_dt.year, snapshot_dt.month, 1)
    next_month = datetime(first_of_month.year + (1 if first_of_month.month == 12 else 0),
                          1 if first_of_month.month == 12 else first_of_month.month + 1, 1)

    candidates = []
    if dataset_name == "clickstream":
        candidates = ["event_ts", "event_date", "snapshot_date"]
    elif dataset_name == "financials":
        candidates = ["txn_ts", "txn_date", "snapshot_date"]
    elif dataset_name == "attributes":
        candidates = ["signup_date", "snapshot_date"]
    else:
        candidates = ["snapshot_date"]

    for c in candidates:
        if c in df.columns:
            # If timestamp/date, do month filter; if string, try cast
            col_ts = F.to_timestamp(col(c))
            df = df.withColumn("_tmp_ts", col_ts)
            df = df.filter((col("_tmp_ts") >= F.lit(first_of_month)) & (col("_tmp_ts") < F.lit(next_month))).drop("_tmp_ts")
            return df

    return df  # no time col found → keep as-is

def _bronze_features_one(dataset_name: str, src_path: str, snapshot_date_str: str,
                         bronze_features_dir: str, spark):
    snap_ym = _ym(snapshot_date_str)
    ingest_dt = datetime.strptime(snapshot_date_str, "%Y-%m-%d").date()

    # Read permissively, schema-on-read
    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .option("mode", "PERMISSIVE")
        .option("quote", '"')
        .option("escape", '"')
        .option("multiLine", True)
        .csv(src_path)
    )

    # Optional month filtering (only if a relevant time column is present)
    df = _filter_to_snapshot_if_possible(df, snapshot_date_str, dataset_name)

    # Add minimal lineage columns
    df = (
        df
        .withColumn("ingest_dt", F.lit(ingest_dt))
        .withColumn("snap_ym", F.lit(snap_ym))
        .withColumn("source_file", F.lit(os.path.basename(src_path)))
    )

    out_dir = os.path.join(bronze_features_dir, dataset_name)
    _ensure_dir(out_dir)
    out_csv = _partition_csv_path(out_dir, f"bronze_{dataset_name}", snapshot_date_str)

    # Save as CSV (to match your LMS Bronze convention)
    df.toPandas().to_csv(out_csv, index=False)
    print(f"[BRONZE][{dataset_name}] rows={df.count()} saved to: {out_csv}")

def process_bronze_features(snapshot_date_str: str, bronze_features_dir: str, raw_dir: str, spark):
    """
    Bronze ingest for feature datasets (raw-as-is + lineage) → CSV per month per dataset.
    """
    _ensure_dir(bronze_features_dir)

    clickstream_path = _find_file(raw_dir, ["feature_clickstream.csv"])
    attributes_path  = _find_file(raw_dir, ["feature_attributes.csv", "features_attributes.csv"])
    financials_path  = _find_file(raw_dir, ["feature_financials.csv", "features_financials.csv"])

    datasets = {
        "clickstream": clickstream_path,
        "attributes":  attributes_path,
        "financials":  financials_path,
    }

    for name, path in datasets.items():
        if path is None:
            print(f"[BRONZE][WARN] Missing source for {name}. Skipping.")
            continue
        _bronze_features_one(name, path, snapshot_date_str, bronze_features_dir, spark)

# -----------------------------
# 3) One-call convenience
# -----------------------------
def process_bronze_all(snapshot_date_str: str,
                       bronze_features_dir: str,
                       bronze_lms_directory: str,
                       raw_dir: str,
                       spark):
    """
    Convenience wrapper to build BOTH feature Bronze and LMS Bronze for a snapshot_date_str.
    """
    process_bronze_features(snapshot_date_str, bronze_features_dir, raw_dir, spark)
    process_bronze_lms(snapshot_date_str, bronze_lms_directory, spark)
