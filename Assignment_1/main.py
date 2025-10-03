import argparse
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

from utils.data_processing_bronze_all import process_bronze_all_monthly_csv
# Silver processors
from utils.data_processing_silver_attributes import process_silver_attributes
from utils.data_processing_silver_clickstream import process_silver_clickstream
from utils.data_processing_silver_financials import process_silver_financials

#  existing label silver 
from utils.data_processing_silver_table import process_silver_table as process_silver_label

BRONZE_FEATURES_DIR = "datamart/bronze/features/"
BRONZE_LMS_DIR      = "datamart/bronze/lms/"
RAW_DIR             = "data"

SILVER_FEATURES_DIR   = "datamart/silver/features/"
SILVER_LMS_DIR        = "datamart/silver/loan_daily/"

def month_range(start_str: str, end_str: str):
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end   = datetime.strptime(end_str,   "%Y-%m-%d")
    cur = datetime(start.year, start.month, 1)
    months = []
    while cur <= end:
        months.append(cur.strftime("%Y-%m-%d"))
        cur = cur + relativedelta(months=1)
    return months

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2023-01-01", help="first-of-month inclusive (YYYY-MM-DD)")
    parser.add_argument("--end",   default="2024-12-01", help="first-of-month inclusive (YYYY-MM-DD)")
    parser.add_argument("--run_bronze", action="store_true", help="also (re)build Bronze for the range")
    args = parser.parse_args()

    spark = (
        pyspark.sql.SparkSession.builder
        .appName("MLE-Silver")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    months = month_range(args.start, args.end)
    print("[INFO] Months to process:", months)

    for ds in months:
        print(f"\n[RUN] Snapshot {ds}")

        # (Optional) Bronze backfill first
        if args.run_bronze:
            process_bronze_all_monthly_csv(
                snapshot_date_str=ds,
                bronze_features_dir=BRONZE_FEATURES_DIR,
                bronze_lms_directory=BRONZE_LMS_DIR,
                raw_dir=RAW_DIR,
                spark=spark
            )

        # ---- Silver: attributes (features) ----
        process_silver_attributes(
            snapshot_date_str=ds,
            bronze_attributes_directory=os.path.join(BRONZE_FEATURES_DIR, "attributes/"),
            silver_attributes_directory=os.path.join(SILVER_FEATURES_DIR, "attributes/"),
            spark=spark
        )

        # ---- Silver: label (lms_loan_daily) ----
        process_silver_label(
            snapshot_date_str=ds,
            bronze_lms_directory=BRONZE_LMS_DIR,
            silver_loan_daily_directory=SILVER_LMS_DIR,
            spark=spark
        )

         # ---- Silver: clickstream (features) ----
        process_silver_clickstream(
            snapshot_date_str=ds,
            bronze_clickstream_directory=os.path.join(BRONZE_FEATURES_DIR, "clickstream/"),
            silver_clickstream_directory=os.path.join(SILVER_FEATURES_DIR, "clickstream/"),
            spark=spark
        )
        # ---- Silver: financials (features) ----
        process_silver_financials(
            snapshot_date_str=ds,
            bronze_financials_directory=os.path.join(BRONZE_FEATURES_DIR, "financials/"),
            silver_financials_directory=os.path.join(SILVER_FEATURES_DIR, "financials/"),
            spark=spark
        )


    spark.stop()
    print("\n[DONE] Silver build complete.")








    


    