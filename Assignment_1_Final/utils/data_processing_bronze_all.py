# utils/data_processing_bronze_all.py
# Bronze monthly CSVs for: clickstream, attributes, financials, lms_loan_daily
# - One CSV per month per dataset, file name includes YYYY_MM_DD
# - Minimal lineage columns on features: ingest_dt, snap_ym, source_file


import os
import glob
import shutil
from datetime import datetime
import pyspark.sql.functions as F
from pyspark.sql.functions import col
def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _ym(date_str: str) -> str:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return f"{dt.year:04d}-{dt.month:02d}"

def _ymd(date_str: str) -> str:
    return date_str.replace("-", "_")

def _find_file(raw_dir: str, candidates):
    for n in candidates:
        p = os.path.join(raw_dir, n)
        if os.path.exists(p):
            return p
    return None

def _write_single_csv_spark(df, out_csv: str):
    tmp_dir = out_csv + "_tmpdir"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    (df.coalesce(1)
       .write.mode("overwrite")
       .option("header", True)
       .csv(tmp_dir))
    part = glob.glob(os.path.join(tmp_dir, "part-*.csv"))
    _ensure_dir(os.path.dirname(out_csv))
    if part:
        shutil.move(part[0], out_csv)
    shutil.rmtree(tmp_dir, ignore_errors=True)

# ---------- unified snapshot filter ----------
def _filter_by_snapshot_date(df, snapshot_date_str: str):
    """
    Keep rows that belong to the same month as snapshot_date_str.
    Works with either:
      - 'snap_ym' column (e.g., '2023-01' or timestamp)
      - 'snapshot_date' column (any day in the month)
    """
    target_ym = _ym(snapshot_date_str)  # 'YYYY-MM'

    if "snap_ym" in df.columns:
        ym_col = F.when(F.length(F.col("snap_ym")) == 7, F.col("snap_ym")) \
                  .otherwise(F.date_format(F.to_timestamp("snap_ym"), "yyyy-MM"))
        return df.filter(ym_col == F.lit(target_ym))

    if "snapshot_date" in df.columns:
        # Try the formats your files actually use (M/D/YYYY), then safe fallbacks
        sd = F.coalesce(
        F.to_date(F.col("snapshot_date"), "d/M/yy"),
        F.to_date(F.col("snapshot_date"), "dd/MM/yy"),
        F.to_date(F.col("snapshot_date"), "d/M/yyyy"),
        F.to_date(F.col("snapshot_date"), "dd/MM/yyyy"),
        F.to_date(F.col("snapshot_date"), "M/d/yyyy"),   # fallback if file is month-first
        F.to_date(F.col("snapshot_date"), "yyyy-MM-dd"), # ISO fallback
        F.to_date(F.col("snapshot_date"))                # last resort
        )
        return df.filter(F.date_format(sd, "yyyy-MM") == F.lit(target_ym))

    raise ValueError("Expected a 'snap_ym' or 'snapshot_date' column for monthly filtering.")


# ---------- LMS (labels source) ----------
def process_bronze_lms_monthly_csv(snapshot_date_str: str, bronze_lms_directory: str, spark):
    _ensure_dir(bronze_lms_directory)
    src = "data/lms_loan_daily.csv"

    df = (spark.read.option("header", True).option("inferSchema", False).csv(src))
    df = _filter_by_snapshot_date(df, snapshot_date_str)

    # lineage
    df = (df
          .withColumn("ingest_dt", F.current_date())
          .withColumn("snap_ym", F.lit(_ym(snapshot_date_str)))
          .withColumn("source_file", F.lit(os.path.basename(src))))

    print(f"[BRONZE][lms] {snapshot_date_str} rows={df.count()}")
    out_csv = os.path.join(bronze_lms_directory, f"bronze_loan_daily_{_ymd(snapshot_date_str)}.csv")
    _write_single_csv_spark(df, out_csv)
    print(f"[BRONZE][lms] saved to: {out_csv}")

# ---------- Features (clickstream / attributes / financials) ----------
def _bronze_features_one_month(dataset_name: str, src_path: str, snapshot_date_str: str,
                               bronze_features_dir: str, spark):
    df = (spark.read
              .option("header", True)
              .option("inferSchema", True)
              .option("mode", "PERMISSIVE")
              .option("quote", '"').option("escape", '"')
              .option("multiLine", True)
              .csv(src_path))

    df = _filter_by_snapshot_date(df, snapshot_date_str)

    # lineage
    df = (df
          .withColumn("ingest_dt", F.current_date())
          .withColumn("snap_ym", F.lit(_ym(snapshot_date_str)))
          .withColumn("source_file", F.lit(os.path.basename(src_path))))

    out_csv = os.path.join(
        bronze_features_dir,
        dataset_name,
        f"bronze_{dataset_name}_{_ymd(snapshot_date_str)}.csv"
    )
    print(f"[BRONZE][{dataset_name}] {snapshot_date_str} rows={df.count()}")
    _ensure_dir(os.path.dirname(out_csv))
    _write_single_csv_spark(df, out_csv)
    print(f"[BRONZE][{dataset_name}] saved to: {out_csv}")

def process_bronze_features_monthly_csv(snapshot_date_str: str, bronze_features_dir: str, raw_dir: str, spark):
    _ensure_dir(bronze_features_dir)
    clickstream_path = _find_file(raw_dir, ["feature_clickstream.csv"])
    attributes_path  = _find_file(raw_dir, ["feature_attributes.csv", "features_attributes.csv"])
    financials_path  = _find_file(raw_dir, ["feature_financials.csv", "features_financials.csv"])

    datasets = {
        "clickstream": clickstream_path,
        "attributes":  attributes_path,
        "financials":  financials_path,
    }
    for name, src in datasets.items():
        if not src:
            print(f"[BRONZE][WARN] Missing source for {name}. Skipping.")
            continue
        _bronze_features_one_month(name, src, snapshot_date_str, bronze_features_dir, spark)

def process_bronze_all_monthly_csv(snapshot_date_str: str,
                                   bronze_features_dir: str,
                                   bronze_lms_directory: str,
                                   raw_dir: str,
                                   spark):
    process_bronze_features_monthly_csv(snapshot_date_str, bronze_features_dir, raw_dir, spark)
    process_bronze_lms_monthly_csv(snapshot_date_str, bronze_lms_directory, spark)