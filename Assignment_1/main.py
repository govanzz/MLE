# main.py — Assignment 1 Orchestrator (Bronze -> Silver -> Gold)
import os
import glob
import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pyspark
from pyspark.sql import functions as F

# ----- Bronze (features + LMS) -----
from utils.data_processing_bronze_all import process_bronze_all_monthly_csv

# ----- Silver processors -----
from utils.data_processing_silver_attributes import process_silver_attributes
from utils.data_processing_silver_clickstream import process_silver_clickstream
from utils.data_processing_silver_financials import process_silver_financials
from utils.data_processing_silver_table import process_silver_table as process_silver_label  # LMS -> Silver

# ----- Gold processors -----
from utils.data_processing_gold_financials import process_gold_financials
from utils.data_processing_gold_attributes import process_gold_attributes
from utils.data_processing_gold_table import process_labels_gold_table          # Label store (Lab 2 util)
from utils.data_processing_gold_model_dataset import process_gold_model_dataset # Final X+Y join

# ----------------- Config paths -----------------
RAW_DIR              = "data"

BRONZE_FEATURES_DIR  = "datamart/bronze/features/"
BRONZE_LMS_DIR       = "datamart/bronze/lms/"

SILVER_FEATURES_DIR  = "datamart/silver/features/"
SILVER_LMS_DIR       = "datamart/silver/loan_daily/"

GOLD_DIR             = "datamart/gold/"
GOLD_FIN_FEAT_DIR    = os.path.join(GOLD_DIR, "features/financials")
GOLD_FIN_WIDE_DIR    = os.path.join(GOLD_DIR, "features/financials_wide")
GOLD_ATTR_FEAT_DIR   = os.path.join(GOLD_DIR, "features/attributes")
GOLD_ATTR_WIDE_DIR   = os.path.join(GOLD_DIR, "features/attributes_wide")
GOLD_LABEL_DIR       = os.path.join(GOLD_DIR, "label_store")
GOLD_MODEL_DIR       = os.path.join(GOLD_DIR, "model_dataset")

# ----------------- Helpers -----------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def month_range(start_str: str, end_str: str):
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end   = datetime.strptime(end_str,   "%Y-%m-%d")
    cur = datetime(start.year, start.month, 1)
    out = []
    while cur <= end:
        out.append(cur.strftime("%Y-%m-%d"))
        cur += relativedelta(months=1)
    return out

# ----------------- Main -----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01-01", help="first-of-month inclusive (YYYY-MM-DD)")
    ap.add_argument("--end",   default="2024-12-01", help="first-of-month inclusive (YYYY-MM-DD)")
    ap.add_argument("--skip_bronze", action="store_true", help="(re)build Bronze from CSV for the range")
    ap.add_argument("--skip_clickstream", action="store_true", help="skip Silver clickstream step")
    ap.add_argument("--dpd", type=int, default=30, help="label DPD threshold (e.g., 30)")
    ap.add_argument("--mob", type=int, default=6,  help="label MOB (e.g., 6)")
    args = ap.parse_args()

    # Spark
    spark = (
        pyspark.sql.SparkSession.builder
        .appName("MLE-Assignment1")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    # Ensure directories exist
    for d in [
        BRONZE_FEATURES_DIR, BRONZE_LMS_DIR,
        SILVER_FEATURES_DIR, SILVER_LMS_DIR,
        GOLD_FIN_FEAT_DIR, GOLD_FIN_WIDE_DIR,
        GOLD_ATTR_FEAT_DIR, GOLD_ATTR_WIDE_DIR,
        GOLD_LABEL_DIR, GOLD_MODEL_DIR
    ]:
        ensure_dir(d)

    months = month_range(args.start, args.end)
    print("[INFO] Months to process:", months)

    for ds in months:
        print(f"\n[RUN] Snapshot {ds}")

        # -------- Bronze (optional) --------
        if not args.skip_bronze:
            process_bronze_all_monthly_csv(
                snapshot_date_str=ds,
                bronze_features_dir=BRONZE_FEATURES_DIR,
                bronze_lms_directory=BRONZE_LMS_DIR,
                raw_dir=RAW_DIR,
                spark=spark
            )

        # -------- Silver --------
        # Attributes
        process_silver_attributes(
            snapshot_date_str=ds,
            bronze_attributes_directory=os.path.join(BRONZE_FEATURES_DIR, "attributes/"),
            silver_attributes_directory=os.path.join(SILVER_FEATURES_DIR, "attributes/"),
            spark=spark
        )

        # LMS -> loan_daily Silver (label source)
        process_silver_label(
            snapshot_date_str=ds,
            bronze_lms_directory=BRONZE_LMS_DIR,
            silver_loan_daily_directory=SILVER_LMS_DIR,
            spark=spark
        )

        # Clickstream (kept in Silver for completeness; no Gold usage)
        if not args.skip_clickstream:
            process_silver_clickstream(
                snapshot_date_str=ds,
                bronze_clickstream_directory=os.path.join(BRONZE_FEATURES_DIR, "clickstream/"),
                silver_clickstream_directory=os.path.join(SILVER_FEATURES_DIR, "clickstream/"),
                spark=spark
            )

        # Financials
        process_silver_financials(
            snapshot_date_str=ds,
            bronze_financials_directory=os.path.join(BRONZE_FEATURES_DIR, "financials/"),
            silver_financials_directory=os.path.join(SILVER_FEATURES_DIR, "financials/"),
            spark=spark
        )

        # -------- Gold: Features --------
        process_gold_financials(
            snapshot_date_str=ds,
            silver_financials_directory=os.path.join(SILVER_FEATURES_DIR, "financials/"),
            gold_financials_features_dir=GOLD_FIN_FEAT_DIR,
            gold_financials_wide_dir=GOLD_FIN_WIDE_DIR,
            spark=spark
        )

        process_gold_attributes(
            snapshot_date_str=ds,
            silver_attributes_directory=os.path.join(SILVER_FEATURES_DIR, "attributes/"),
            gold_attributes_features_dir=GOLD_ATTR_FEAT_DIR,
            gold_attributes_wide_dir=GOLD_ATTR_WIDE_DIR,
            spark=spark
        )

        # -------- Gold: Label Store --------
        process_labels_gold_table(
            snapshot_date_str=ds,
            silver_loan_daily_directory=SILVER_LMS_DIR,
            gold_label_store_directory=GOLD_LABEL_DIR,
            spark=spark,
            dpd=args.dpd,
            mob=args.mob
        )

        # -------- Gold: Final Model Dataset (Label ⟕ Fin ⟕ Attr) --------
        process_gold_model_dataset(
            snapshot_date_str=ds,
            gold_label_store_dir=GOLD_LABEL_DIR,
            gold_financials_features_dir=GOLD_FIN_FEAT_DIR,
            gold_attributes_features_dir=GOLD_ATTR_FEAT_DIR,
            out_dir=GOLD_MODEL_DIR,
            spark=spark
        )

    # -------- Sanity check across months --------
    md_paths = sorted(glob.glob(os.path.join(GOLD_MODEL_DIR, "gold_model_dataset_*.parquet")))
    if md_paths:
        md_all = spark.read.parquet(*md_paths)
        print(f"\n[CHECK] model_dataset rows={md_all.count():,} cols={len(md_all.columns)}")
        md_all.groupBy("snapshot_date").count().orderBy("snapshot_date").show(30, False)

    spark.stop()
    print("\n[DONE] Silver + Gold build complete.")








    


    