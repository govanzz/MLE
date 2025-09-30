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

from utils.data_processing_bronze_all import process_bronze_all_single_parquet
from utils.feature_engineering_clickstream import run_clickstream_minimal_fe  # <-- uses NON-winsorized util

BRONZE_FEATURES_DIR = "datamart/bronze/features/"
BRONZE_LMS_DIR      = "datamart/bronze/lms/"
RAW_DIR             = "data"

SILVER_FEATURES_DIR = "datamart/silver/features/"
SILVER_CLICKSTREAM_PARQUET = os.path.join(SILVER_FEATURES_DIR, "user_clickstream_features")
SILVER_CLICKSTREAM_PREVIEW = os.path.join(SILVER_FEATURES_DIR, "user_clickstream_features_preview")

if __name__ == "__main__":
    spark = (pyspark.sql.SparkSession.builder
             .appName("dev")
             .master("local[*]")
             .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    # 1) Bronze ingest
    process_bronze_all_single_parquet(
        bronze_features_dir=BRONZE_FEATURES_DIR,
        bronze_lms_directory=BRONZE_LMS_DIR,
        raw_dir=RAW_DIR,
        spark=spark
    )

    # 2) Silver: per-customer aggregates (NO winsorization)
    bronze_clickstream_dir = os.path.join(BRONZE_FEATURES_DIR, "clickstream")
    meta = run_clickstream_minimal_fe(
        spark=spark,
        bronze_clickstream_dir=bronze_clickstream_dir,
        silver_out_parquet=SILVER_CLICKSTREAM_PARQUET,
        silver_preview_csv_dir=SILVER_CLICKSTREAM_PREVIEW
    )
    print("[SILVER][META]", meta)

    spark.stop()








    


    