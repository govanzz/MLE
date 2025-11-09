# utils/feature_engineering_clickstream.py
import os
import glob
import pyspark.sql.functions as F

def _print_schema_sample(df):
    print("=== SCHEMA ==="); df.printSchema()
    print("=== COLUMNS ===", df.columns)
    print("=== SAMPLE ==="); df.show(10, truncate=False)

def _write_single_csv(df, out_dir: str):
    """
    Writes a 1-file CSV preview and returns the part file path.
    """
    (df.coalesce(1)
       .write.mode("overwrite")
       .option("header", True)
       .csv(out_dir))
    parts = glob.glob(os.path.join(out_dir, "part-*.csv"))
    path = parts[0] if parts else out_dir
    print(f"[SILVER][CSV PREVIEW] {path}")
    return path

def run_clickstream_minimal_fe(
    spark,
    bronze_clickstream_dir: str = "datamart/bronze/features/clickstream",
    silver_out_parquet: str     = "datamart/silver/features/user_clickstream_features",
    silver_preview_csv_dir: str = "datamart/silver/features/user_clickstream_features_preview"
):
    """
    Reads Bronze clickstream parquet and produces a per-customer Silver table
    WITHOUT any winsorization/clipping.

    Aggregates per fe_*: mean, median (approx), std, min, max, sum, missing_rate
    Plus coverage: n_rows, first_snapshot, last_snapshot, days_active
    """
    ID_COL   = "Customer_ID"
    DATE_COL = "snapshot_date"

    # 1) Load Bronze
    df = spark.read.parquet(bronze_clickstream_dir)
    _print_schema_sample(df)

    # 2) Roles & types
    fe_cols = [c for c in df.columns if c.startswith("fe_")]
    if DATE_COL in df.columns:
        df = df.withColumn(DATE_COL, F.to_timestamp(DATE_COL))
    for c in fe_cols:
        df = df.withColumn(c, F.col(c).cast("double"))

    # 3) Per-customer aggregates (no clipping)
    gb = df.groupBy(ID_COL)

    # Coverage
    basic = gb.agg(
        F.count(F.lit(1)).alias("n_rows"),
        F.min(DATE_COL).alias("first_snapshot") if DATE_COL in df.columns else F.first(ID_COL).alias("first_snapshot"),
        F.max(DATE_COL).alias("last_snapshot")  if DATE_COL in df.columns else F.first(ID_COL).alias("last_snapshot"),
    )
    if DATE_COL in df.columns:
        basic = basic.withColumn(
            "days_active",
            F.when(
                F.col("first_snapshot").isNotNull() & F.col("last_snapshot").isNotNull(),
                (F.col("last_snapshot").cast("long") - F.col("first_snapshot").cast("long")) / 86400.0
            )
        )

    # Feature stats
    aggs = []
    for c in fe_cols:
        aggs += [
            F.avg(c).alias(f"{c}_mean"),
            F.expr(f"percentile_approx({c}, 0.5)").alias(f"{c}_median"),
            F.stddev_samp(c).alias(f"{c}_std"),
            F.min(c).alias(f"{c}_min"),
            F.max(c).alias(f"{c}_max"),
            F.sum(c).alias(f"{c}_sum"),
            F.avg(F.col(c).isNull().cast("double")).alias(f"{c}_missing_rate"),
        ]
    feat_agg = gb.agg(*aggs)

    silver = basic.join(feat_agg, on=ID_COL, how="left")

    # 4) Write Silver parquet
    (silver.write.mode("overwrite").parquet(silver_out_parquet))
    print(f"[SILVER][OK] wrote Parquet → {silver_out_parquet}")

    # 5) Small CSV preview (top 200 by n_rows)
    preview_df = silver.orderBy(F.desc("n_rows")).limit(200)
    preview_file = _write_single_csv(preview_df, silver_preview_csv_dir)

    return {
        "rows": silver.count(),
        "cols": len(silver.columns),
        "parquet_out": silver_out_parquet,
        "csv_preview": preview_file,
    }
