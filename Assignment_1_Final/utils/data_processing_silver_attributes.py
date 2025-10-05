# utils/data_processing_silver_attributes.py

import os
from datetime import datetime
import pyspark.sql.functions as F
from pyspark.sql.types import StringType, DateType

def process_silver_attributes(snapshot_date_str, bronze_attributes_directory, silver_attributes_directory, spark):
    """
    Reads Bronze attributes CSV for the given snapshot_date, drops PII/noisy fields,
    standardizes occupation, and writes Silver Parquet.
    Input  : bronze_attributes_<YYYY_MM_DD>.csv
    Output : silver_attributes_<YYYY_MM_DD>.parquet
    """
    # read Bronze
    in_name = f"bronze_attributes_{snapshot_date_str.replace('-', '_')}.csv"
    in_path = os.path.join(bronze_attributes_directory, in_name)
    df = spark.read.csv(in_path, header=True, inferSchema=True)
    print("[ATTR] loaded:", in_path, "| rows:", df.count())

    # transform  keep only: customer_id, occupation (cleaned), snapshot_date, lineage
    df_silver = (
        df
        .withColumn("customer_id", F.col("Customer_ID").cast(StringType()))
        .withColumn(
            "occupation",
            F.when(F.col("Occupation") == "_______", None)
             .otherwise(F.upper(F.trim(F.col("Occupation"))))
        )
        .withColumn("snapshot_date", F.col("snapshot_date").cast(DateType()))
        .select(
            "customer_id",
            "occupation",
            "snapshot_date",
            "ingest_dt",      # lineage
            "snap_ym",
            "source_file"
        )
    )

    # write Silver (Parquet)
    out_name = f"silver_attributes_{snapshot_date_str.replace('-', '_')}.parquet"
    out_path = os.path.join(silver_attributes_directory, out_name)
    df_silver.write.mode("overwrite").parquet(out_path)
    print("[ATTR] saved :", out_path, "| rows:", df_silver.count())
    return df_silver


