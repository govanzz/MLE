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

from utils.data_processing_bronze_all import process_bronze_all_single_parquet

BRONZE_FEATURES_DIR = "datamart/bronze/features/"
BRONZE_LMS_DIR      = "datamart/bronze/lms/"
RAW_DIR             = "data"

if __name__ == "__main__":
    spark = (pyspark.sql.SparkSession.builder
             .appName("dev")
             .master("local[*]")
             # optional: consistent behavior if you later partition in Silver/Gold
             .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    process_bronze_all_single_parquet(
        bronze_features_dir=BRONZE_FEATURES_DIR,
        bronze_lms_directory=BRONZE_LMS_DIR,
        raw_dir=RAW_DIR,
        spark=spark
    )

    spark.stop()







    


    