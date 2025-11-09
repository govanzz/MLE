# utils/data_processing_silver_clickstream.py
import os
from datetime import datetime
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StringType, DateType, DoubleType

def process_silver_clickstream(snapshot_date_str, bronze_clickstream_directory, silver_clickstream_directory, spark):
    """
    Reads Bronze clickstream CSV for the given snapshot_date, casts fe_* to double,
    standardizes keys, dedupes by (customer_id, snapshot_date) keeping latest ingest_dt,
    and writes Silver Parquet.
      IN : bronze_clickstream_<YYYY_MM_DD>.csv
      OUT: silver_clickstream_<YYYY_MM_DD>.parquet
    """
    # --- load Bronze (single month) ---
    in_name = f"bronze_clickstream_{snapshot_date_str.replace('-', '_')}.csv"
    in_path = os.path.join(bronze_clickstream_directory, in_name)
    df = spark.read.csv(in_path, header=True, inferSchema=True)
    print("[CLK] loaded:", in_path, "| rows:", df.count())

    # --- standardize columns / types ---
    # keys & dates
    df = (
        df
        .withColumn("customer_id", F.col("Customer_ID").cast(StringType()))
        .withColumn("snapshot_date", F.col("snapshot_date").cast(DateType()))
    )

    # cast all fe_* columns to double
    fe_cols = [c for c in df.columns if c.lower().startswith("fe_")]
    for c in fe_cols:
        df = df.withColumn(c, F.col(c).cast(DoubleType()))

    # keep only features + keys + lineage
    keep_cols = ["customer_id", "snapshot_date", "ingest_dt", "snap_ym", "source_file"] + fe_cols
    df = df.select(*keep_cols)

    # --- dedupe: (customer_id, snapshot_date) keep latest ingest_dt ---
    if "ingest_dt" in df.columns:
        w = Window.partitionBy("customer_id", "snapshot_date").orderBy(F.col("ingest_dt").desc_nulls_last())
        df = (
            df.withColumn("_rn", F.row_number().over(w))
              .filter(F.col("_rn") == 1)
              .drop("_rn")
        )
    else:
        df = df.dropDuplicates(["customer_id", "snapshot_date"])

    # --- write Silver (Parquet) ---
    out_name = f"silver_clickstream_{snapshot_date_str.replace('-', '_')}.parquet"
    out_path = os.path.join(silver_clickstream_directory, out_name)
    df.write.mode("overwrite").parquet(out_path)
    print("[CLK] saved :", out_path, "| rows:", df.count())
    return df
