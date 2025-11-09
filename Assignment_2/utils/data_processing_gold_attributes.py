# utils/data_processing_gold_attributes.py
from datetime import datetime
from pyspark.sql import functions as F

# Fixed vocabulary (from your audit: 15 categories)
OCCUPATION_VOCAB = [
    "ACCOUNTANT","ARCHITECT","DEVELOPER","DOCTOR","ENGINEER",
    "ENTREPRENEUR","JOURNALIST","LAWYER","MANAGER","MEDIA_MANAGER",
    "MECHANIC","MUSICIAN","SCIENTIST","TEACHER","WRITER"
]

def _token(snapshot_date_str: str) -> str:
    return datetime.strptime(snapshot_date_str, "%Y-%m-%d").strftime("%Y_%m_%d")

def process_gold_attributes(
    snapshot_date_str: str,
    silver_attributes_directory: str,
    gold_attributes_features_dir: str,
    gold_attributes_wide_dir: str,
    spark,
):
    """
    Reads:  {silver_attributes_directory}/silver_attributes_<YYYY_MM_DD>.parquet
    Writes: {gold_attributes_features_dir}/gold_attributes_features_<YYYY_MM_DD>.parquet
            {gold_attributes_wide_dir}/gold_attributes_wide_<YYYY_MM_DD>.parquet

    Spec: OHE over fixed 15-category vocab + is_occupation_null flag.
    """
    tok = _token(snapshot_date_str)
    in_path  = f"{silver_attributes_directory}/silver_attributes_{tok}.parquet"
    out_feat = f"{gold_attributes_features_dir}/gold_attributes_features_{tok}.parquet"
    out_wide = f"{gold_attributes_wide_dir}/gold_attributes_wide_{tok}.parquet"

    df = spark.read.parquet(in_path)

    # Normalize: upper+trim; literal "NULL" or "" -> actual null
    occ_up = F.upper(F.trim(F.col("occupation")))
    occ_clean = F.when(occ_up.isNull() | occ_up.isin("NULL", ""), None).otherwise(occ_up)

    d = (df
         .withColumn("occupation_norm", occ_clean)
         .withColumn("is_occupation_null", F.col("occupation_norm").isNull().cast("int"))
    )

    # One-hot for fixed vocab (consistent schema across months)
    for cat in OCCUPATION_VOCAB:
        colname = f"occ_{cat}"
        d = d.withColumn(colname, F.when(F.col("occupation_norm") == F.lit(cat), 1).otherwise(0))

    # FEATURES subset (keys + OHE + null flag)
    keys = ["customer_id", "snapshot_date", "snap_ym"]
    ohe_cols = [f"occ_{c}" for c in OCCUPATION_VOCAB]
    feat_cols = keys + ["is_occupation_null"] + ohe_cols
    d.select(*feat_cols).write.mode("overwrite").parquet(out_feat)

    # WIDE (audit): keep raw + engineered
    d.drop("occupation_norm").write.mode("overwrite").parquet(out_wide)

    print(f"[GOLD][attributes_features] wrote: {out_feat}")
    print(f"[GOLD][attributes_wide]     wrote: {out_wide}")
