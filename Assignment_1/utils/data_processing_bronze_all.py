# utils/data_processing_bronze_all.py
# Combined Bronze ingest for: LMS (labels source) + Feature datasets

# --- SINGLE-TABLE BRONZE  PARQUET --------------------
import os
from datetime import datetime
import pyspark.sql.functions as F

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _find_file(raw_dir: str, candidates):
    for name in candidates:
        p = os.path.join(raw_dir, name)
        if os.path.exists(p):
            return p
    return None

def _write_parquet_overwrite(df, out_dir: str):
    _ensure_dir(out_dir)
    # No partitioning; single logical table as a Parquet dataset directory
    (df.write
       .mode("overwrite")
       .parquet(out_dir))
    print(f"[BRONZE][OK] wrote PARQUET: {out_dir}")

def process_bronze_features_single_parquet(bronze_features_dir: str, raw_dir: str, spark):
    """
    Writes ONE Parquet dataset per feature:
      datamart/bronze/features/clickstream/
      datamart/bronze/features/attributes/
      datamart/bronze/features/financials/
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

    for name, src in datasets.items():
        if not src:
            print(f"[BRONZE][WARN] Missing source for {name}. Skipping.")
            continue

        print(f"[BRONZE][INFO] Ingesting {name} from {src}")
        df = (spark.read
                 .option("header", True)
                 .option("inferSchema", True)
                 .option("mode", "PERMISSIVE")
                 .option("quote", '"')
                 .option("escape", '"')
                 .option("multiLine", True)
                 .csv(src)
             ) \
             .withColumn("ingest_dt", F.current_date()) \
             .withColumn("source_file", F.lit(os.path.basename(src)))

        out_dir = os.path.join(bronze_features_dir, name)  # e.g., .../features/clickstream
        _write_parquet_overwrite(df, out_dir)
        print(f"[BRONZE][{name}] rows={df.count()} cols={len(df.columns)}")

def process_bronze_lms_single_parquet(bronze_lms_directory: str, raw_dir: str, spark):
    """
    Writes ONE Parquet dataset for LMS:
      datamart/bronze/lms/loan_daily/
    """
    _ensure_dir(bronze_lms_directory)
    src = os.path.join(raw_dir, "lms_loan_daily.csv")
    if not os.path.exists(src):
        print("[BRONZE][WARN] Missing data/lms_loan_daily.csv")
        return

    print(f"[BRONZE][INFO] Ingesting LMS from {src}")
    df = (spark.read
             .option("header", True)
             .option("inferSchema", True)
             .option("mode", "PERMISSIVE")
             .csv(src)
         ) \
         .withColumn("ingest_dt", F.current_date()) \
         .withColumn("source_file", F.lit(os.path.basename(src)))

    out_dir = os.path.join(bronze_lms_directory, "loan_daily")  # e.g., .../bronze/lms/loan_daily/
    _write_parquet_overwrite(df, out_dir)
    print(f"[BRONZE][lms] rows={df.count()} cols={len(df.columns)}")

def process_bronze_all_single_parquet(bronze_features_dir: str, bronze_lms_directory: str, raw_dir: str, spark):
    process_bronze_features_single_parquet(bronze_features_dir, raw_dir, spark)
    process_bronze_lms_single_parquet(bronze_lms_directory, raw_dir, spark)
