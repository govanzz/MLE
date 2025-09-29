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

#import utils.data_processing_bronze_table
#import utils.data_processing_silver_table
#import utils.data_processing_gold_table

from utils.data_processing_bronze_all import process_bronze_all

BRONZE_FEATURES_DIR = "datamart/bronze/features/"
BRONZE_LMS_DIR      = "datamart/bronze/lms/"
RAW_DIR             = "data"

def month_range(start_str: str, end_str: str):
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end   = datetime.strptime(end_str,   "%Y-%m-%d")
    cur = datetime(start.year, start.month, 1)
    months = []
    while cur <= end:
        months.append(cur.strftime("%Y-%m-%d"))
        if cur.month == 12:
            cur = datetime(cur.year + 1, 1, 1)
        else:
            cur = datetime(cur.year, cur.month + 1, 1)
    return months

if __name__ == "__main__":
    # CLI args
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2023-01-01", help="first-of-month inclusive (YYYY-MM-DD)")
    parser.add_argument("--end",   default="2024-12-01", help="first-of-month inclusive (YYYY-MM-DD)")
    args = parser.parse_args()

    # Spark
    spark = pyspark.sql.SparkSession.builder.appName("dev").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # Generate month list
    months = month_range(args.start, args.end)
    print("[INFO] Months to backfill:", months)

    # Run Bronze for each month (features + LMS)
    for ds in months:
        print(f"[RUN] Snapshot {ds}")
        process_bronze_all(
            snapshot_date_str=ds,
            bronze_features_dir=BRONZE_FEATURES_DIR,
            bronze_lms_directory=BRONZE_LMS_DIR,
            raw_dir=RAW_DIR,
            spark=spark
        )

    spark.stop()






    


    