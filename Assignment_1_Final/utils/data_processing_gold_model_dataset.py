# utils/data_processing_gold_model_dataset.py
from datetime import datetime
from pyspark.sql import functions as F

def _token(ds: str) -> str:
    return datetime.strptime(ds, "%Y-%m-%d").strftime("%Y_%m_%d")

def process_gold_model_dataset(
    snapshot_date_str: str,
    gold_label_store_dir: str,
    gold_financials_features_dir: str,
    gold_attributes_features_dir: str,
    out_dir: str,
    spark,
):
    """
    Reads:
      - {gold_label_store_dir}/gold_label_store_<YYYY_MM_DD>.parquet
      - {gold_financials_features_dir}/gold_financials_features_<YYYY_MM_DD>.parquet
      - {gold_attributes_features_dir}/gold_attributes_features_<YYYY_MM_DD>.parquet
    Writes:
      - {out_dir}/gold_model_dataset_<YYYY_MM_DD>.parquet
    Join keys: (customer_id, snapshot_date). Note: label store uses 'Customer_ID' -> normalized.
    """
    tok = _token(snapshot_date_str)
    p_lbl = f"{gold_label_store_dir}/gold_label_store_{tok}.parquet"
    p_fin = f"{gold_financials_features_dir}/gold_financials_features_{tok}.parquet"
    p_att = f"{gold_attributes_features_dir}/gold_attributes_features_{tok}.parquet"
    p_out = f"{out_dir}/gold_model_dataset_{tok}.parquet"

    lbl = (spark.read.parquet(p_lbl)
           .withColumnRenamed("Customer_ID", "customer_id")
           .select("customer_id", "snapshot_date", "loan_id", "label", "label_def"))

    fin = spark.read.parquet(p_fin)
    att = spark.read.parquet(p_att)

    # Avoid duplicate snap_ym columns when joining features
    fin_feats = fin  # keep fin.snap_ym
    att_feats = att.drop("snap_ym")

    ds = (lbl
          .join(fin_feats, ["customer_id", "snapshot_date"], "left")
          .join(att_feats, ["customer_id", "snapshot_date"], "left"))

    ds.write.mode("overwrite").parquet(p_out)
    print(f"[GOLD][model_dataset] wrote: {p_out}")

    return ds
