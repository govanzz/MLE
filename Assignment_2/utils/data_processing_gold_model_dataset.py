# utils/data_processing_gold_model_dataset.py
from datetime import datetime
from pyspark.sql import functions as F, Window

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
    Builds the Gold model dataset with **AS-OF join logic**:
      - For each (customer_id, label snapshot_date),
        use the latest financial & attribute records with feat_date <= label_date.

    Reads:
      - {gold_label_store_dir}/gold_label_store_<YYYY_MM_DD>.parquet
      - All financial feature files in {gold_financials_features_dir}/
      - All attribute feature files in {gold_attributes_features_dir}/
    Writes:
      - {out_dir}/gold_model_dataset_<YYYY_MM_DD>.parquet
    """

    tok = _token(snapshot_date_str)
    p_lbl = f"{gold_label_store_dir}/gold_label_store_{tok}.parquet"
    p_out = f"{out_dir}/gold_model_dataset_{tok}.parquet"

    # ---- Load label store ----
    lbl = (
        spark.read.parquet(p_lbl)
        .withColumnRenamed("Customer_ID", "customer_id")
        .withColumn("customer_id", F.upper(F.trim(F.col("customer_id"))))
        .withColumn("label_date", F.to_date("snapshot_date"))
        .select("customer_id", "label_date", "loan_id", "label", "label_def")
    )

    # ---- Load *all* feature files (multi-month context) ----
    fin = (
        spark.read.parquet(f"{gold_financials_features_dir}/*.parquet")
        .withColumnRenamed("Customer_ID", "customer_id")
        .withColumn("customer_id", F.upper(F.trim(F.col("customer_id"))))
        .withColumn(
            "feat_date",
            F.coalesce(
                F.to_date("snapshot_date"),
                F.to_date(F.to_timestamp("snap_ym"))
            )
        )
    )

    att = (
        spark.read.parquet(f"{gold_attributes_features_dir}/*.parquet")
        .withColumnRenamed("Customer_ID", "customer_id")
        .withColumn("customer_id", F.upper(F.trim(F.col("customer_id"))))
        .withColumn(
            "feat_date",
            F.coalesce(
                F.to_date("snapshot_date"),
                F.to_date(F.to_timestamp("snap_ym"))
            )
        )
    )

    # ---- Feature subsets ----
    fin_keep = [c for c in fin.columns if c not in {"snapshot_date","snap_ym","customer_id","feat_date"}]
    att_keep = [c for c in att.columns if c not in {"snapshot_date","snap_ym","customer_id","feat_date"}]

    # ---- AS-OF join logic ----
    # Financials
    fin_join = (
        lbl.join(F.broadcast(fin.select("customer_id","feat_date", *fin_keep)), "customer_id", "left")
        .where(F.col("feat_date").isNotNull() & (F.col("feat_date") <= F.col("label_date")))
    )
    w_fin = Window.partitionBy("customer_id","label_date").orderBy(F.col("feat_date").desc())
    fin_ranked = fin_join.withColumn("rn", F.row_number().over(w_fin)).where(F.col("rn")==1).drop("rn")

    # Attributes
    att_join = (
        lbl.join(F.broadcast(att.select("customer_id","feat_date", *att_keep)), "customer_id", "left")
        .where(F.col("feat_date").isNotNull() & (F.col("feat_date") <= F.col("label_date")))
    )
    w_att = Window.partitionBy("customer_id","label_date").orderBy(F.col("feat_date").desc())
    att_ranked = att_join.withColumn("rn", F.row_number().over(w_att)).where(F.col("rn")==1).drop("rn")

    # ---- Merge back to form Gold model dataset ----
    base_keys = ["customer_id", "label_date", "loan_id", "label", "label_def"]
    fin_clean = fin_ranked.drop(*[c for c in base_keys if c not in ["customer_id","label_date"]])
    att_clean = att_ranked.drop(*[c for c in base_keys if c not in ["customer_id","label_date"]])

    ds = (
        lbl
        .join(fin_clean, ["customer_id","label_date"], "left")
        .join(att_clean, ["customer_id","label_date"], "left")
        .drop("feat_date")
        .withColumnRenamed("label_date", "snapshot_date")
    )

    ds.write.mode("overwrite").parquet(p_out)
    print(f"[GOLD][model_dataset_asof] wrote: {p_out}")
    return ds
