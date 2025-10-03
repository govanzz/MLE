# utils/data_processing_silver_financials.py
#
# Reads : {bronze_financials_directory}/bronze_financials_<YYYY_MM_DD>.csv
# Writes: {silver_financials_directory}/silver_financials_<YYYY_MM_DD>.parquet
#
# Silver rules implemented:
# - Keep customer_id; enforce lineage types; trim all strings
# - Money to 2 d.p.: annual_income, outstanding_debt; monthly_inhand_salary/amount_invested_monthly/monthly_balance to 2 d.p.
# - interest_rate => percent (handles fractions, percents, commas, basis-points)
# - num_of_loan => (number of commas in raw type_of_loan) + 1, with NULL/empty -> 0
# - delay_from_due_date int; num_of_delayed_payment parse int from strings like '14_'
# - changed_credit_limit double; num_credit_inquiries int; credit_utilization_ratio double
# - credit_history_age -> months (int): years*12 + months
# - credit_mix normalize (Good/Standard/Bad) and map NULL-like tokens to null
# - payment_behaviour: null-out '!@9#%8' and NULL-like tokens; keep others raw
# - payment_of_min_amount: keep raw
# - Dedup by (customer_id, snapshot_date) keeping latest ingest_dt; idempotent monthly write

from datetime import datetime
from pyspark.sql import Window
import pyspark.sql.functions as F
import pyspark.sql.types as T

# tokens that should be treated as null for categoricals
NULL_TOKENS = ["_", "NULL", "N/A", "NA", ""]

def _normalize_cols_to_snake(df):
    def norm(c): return c.strip().lower().replace(" ", "_")
    return df.toDF(*[norm(c) for c in df.columns])

def _money_to_2dp(col):
    # '52312.68_', '$3,500', '₹1,20,000.0' -> numeric to 2dp
    s = F.regexp_replace(F.col(col).cast("string"), r"[^\d\.\-]", "")
    return F.round(s.cast("double"), 2)

def _plain_int_from_str(col):
    # '14_' -> 14
    s = F.regexp_replace(F.col(col).cast("string"), r"[^0-9]", "")
    return s.cast("int")

def _interest_to_percent(col):
    """
    Accepts: '14%', '14', '0.14', '0.14_', '1,060', '5059'
    Output: percent on 0..100 scale, rounded to 2 d.p.
      <= 1.0          -> fraction -> x100  (0.14 -> 14.00)
      100..10000      -> basis points -> /100 (1,060 -> 10.60, 5059 -> 50.59)
      otherwise       -> already percent
    """
    s = F.regexp_replace(F.col(col).cast("string"), r"[_,\s,%]", "")
    v = s.cast("double")
    pct = (F.when(v.isNull(), None)
             .when(v <= 1.0, v * 100.0)
             .when((v > 100.0) & (v <= 10000.0), v / 100.0)
             .otherwise(v))
    return F.round(pct, 2)

def _credit_history_to_months(col):
    # '15 Years and 10 Months' -> 15*12 + 10 = 190
    txt = F.col(col).cast("string")
    yrs = F.regexp_extract(txt, r"(\d+)\s*Years?", 1)
    mos = F.regexp_extract(txt, r"(\d+)\s*Months?", 1)
    yrs_i = F.when(yrs == "", F.lit(0)).otherwise(yrs.cast("int"))
    mos_i = F.when(mos == "", F.lit(0)).otherwise(mos.cast("int"))
    return (yrs_i * F.lit(12) + mos_i).cast("int")

def process_silver_financials(
    snapshot_date_str: str,
    bronze_financials_directory: str,
    silver_financials_directory: str,
    spark
):
    """
    Reads:  {bronze_financials_directory}/bronze_financials_<YYYY_MM_DD>.csv
    Writes: {silver_financials_directory}/silver_financials_<YYYY_MM_DD>.parquet
    """
    # token for month-specific files
    dt = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    token = dt.strftime("%Y_%m_%d")

    bronze_path = f"{bronze_financials_directory}/bronze_financials_{token}.csv"
    silver_path = f"{silver_financials_directory}/silver_financials_{token}.parquet"

    df = (
        spark.read.option("header", True)
        .option("multiLine", True)
        .csv(bronze_path)
    )

    # Normalize names
    df = _normalize_cols_to_snake(df)

    # Fix expected header typos to canonical names
    if "interst_rate" in df.columns and "interest_rate" not in df.columns:
        df = df.withColumnRenamed("interst_rate", "interest_rate")
    if "payement_behaviour" in df.columns and "payment_behaviour" not in df.columns:
        df = df.withColumnRenamed("payement_behaviour", "payment_behaviour")

    # Cast lineage
    if "ingest_dt" in df.columns:
        df = df.withColumn("ingest_dt", F.to_timestamp("ingest_dt"))
    if "snapshot_date" in df.columns:
        df = df.withColumn("snapshot_date", F.to_date("snapshot_date"))

    # Trim all strings once
    for c, t in df.dtypes:
        if t == "string":
            df = df.withColumn(c, F.trim(F.col(c)))

    # ----------------- CLEANING (per spec) -----------------

    # Annual_Income: remove symbols, 2 d.p.
    if "annual_income" in df.columns:
        df = df.withColumn("annual_income", _money_to_2dp("annual_income"))

    # Monthly_Inhand_Salary: 2 d.p.
    if "monthly_inhand_salary" in df.columns:
        df = df.withColumn("monthly_inhand_salary", F.round(F.col("monthly_inhand_salary").cast("double"), 2))

    # Interest_Rate: percent scale (0..100), 2 d.p.
    if "interest_rate" in df.columns:
        df = df.withColumn("interest_rate", _interest_to_percent("interest_rate"))

    # Num_of_Loan: EXACT rule = number_of_commas + 1 (no normalization of text)
    if "type_of_loan" in df.columns:
        tol_raw = F.col("type_of_loan").cast("string")
        is_zero = (
            tol_raw.isNull() |
            (F.length(tol_raw) == 0) |
            (F.upper(F.trim(tol_raw)) == F.lit("NULL"))
        )
        comma_count = F.length(tol_raw) - F.length(F.regexp_replace(tol_raw, ",", ""))
        df = df.withColumn(
            "num_of_loan",
            F.when(is_zero, F.lit(0)).otherwise((comma_count + F.lit(1)).cast("int"))
        )
        # keep raw type_of_loan as-is

    # Delay_from_due_date: int
    if "delay_from_due_date" in df.columns:
        df = df.withColumn("delay_from_due_date", F.col("delay_from_due_date").cast("int"))

    # Num_of_Delayed_Payment: parse to int (e.g., '14_' -> 14)
    if "num_of_delayed_payment" in df.columns:
        df = df.withColumn("num_of_delayed_payment", _plain_int_from_str("num_of_delayed_payment"))

    # Changed_Credit_Limit: numeric
    if "changed_credit_limit" in df.columns:
        df = df.withColumn("changed_credit_limit", F.col("changed_credit_limit").cast("double"))

    # Num_Credit_Inquiries: int
    if "num_credit_inquiries" in df.columns:
        df = df.withColumn("num_credit_inquiries", F.col("num_credit_inquiries").cast("int"))

    # Credit_Mix: map NULL-like tokens to null; normalize others to Title Case
    if "credit_mix" in df.columns:
        up = F.upper(F.col("credit_mix"))
        df = df.withColumn(
            "credit_mix",
            F.when(up.isin(*NULL_TOKENS), F.lit(None).cast(T.StringType()))
             .otherwise(F.initcap(F.lower(F.col("credit_mix"))))
        )

    # Outstanding_Debt: 2 d.p.
    if "outstanding_debt" in df.columns:
        df = df.withColumn("outstanding_debt", _money_to_2dp("outstanding_debt"))

    # Credit_Utilization_Ratio: double
    if "credit_utilization_ratio" in df.columns:
        df = df.withColumn("credit_utilization_ratio", F.col("credit_utilization_ratio").cast("double"))

    # Credit_History_Age -> months (int)
    if "credit_history_age" in df.columns:
        df = df.withColumn("credit_history_age", _credit_history_to_months("credit_history_age"))

    # Payment_of_Min_Amount: keep as-is (string)
    # Total_EMI_per_month: type-enforce double
    if "total_emi_per_month" in df.columns:
        df = df.withColumn("total_emi_per_month", F.col("total_emi_per_month").cast("double"))

    # Amount_invested_monthly: 2 d.p.
    if "amount_invested_monthly" in df.columns:
        df = df.withColumn("amount_invested_monthly", F.round(F.col("amount_invested_monthly").cast("double"), 2))

    # Payment_Behaviour: null-out junk & null-like tokens, keep others raw
    if "payment_behaviour" in df.columns:
        up = F.upper(F.col("payment_behaviour"))
        df = df.withColumn(
            "payment_behaviour",
            F.when((F.col("payment_behaviour") == "!@9#%8") | up.isin(*NULL_TOKENS),
                   F.lit(None).cast(T.StringType()))
             .otherwise(F.col("payment_behaviour"))
        )

    # Monthly_Balance: 2 d.p.
    if "monthly_balance" in df.columns:
        df = df.withColumn("monthly_balance", F.round(F.col("monthly_balance").cast("double"), 2))

    # ----------------- DEDUP & SELECT -----------------

    # Dedup within (customer_id, snapshot_date) by latest ingest_dt
    if all(c in df.columns for c in ["customer_id", "snapshot_date", "ingest_dt"]):
        w = Window.partitionBy("customer_id", "snapshot_date").orderBy(F.col("ingest_dt").desc_nulls_last())
        df = df.withColumn("_rn", F.row_number().over(w)).filter(F.col("_rn") == 1).drop("_rn")
    else:
        df = df.dropDuplicates(["customer_id", "snapshot_date"])

    # Column order
    wanted = [
        "customer_id", "snapshot_date",
        "ingest_dt", "snap_ym", "source_file",
        "annual_income", "monthly_inhand_salary", "interest_rate",
        "num_of_loan", "delay_from_due_date", "num_of_delayed_payment",
        "changed_credit_limit", "num_credit_inquiries", "credit_mix",
        "outstanding_debt", "credit_utilization_ratio", "credit_history_age",
        "payment_of_min_amount", "total_emi_per_month", "amount_invested_monthly",
        "payment_behaviour", "monthly_balance", "type_of_loan",
    ]
    existing = [c for c in wanted if c in df.columns]
    df_out = df.select(*existing)

    # ----------------- WRITE (idempotent, per-month file) -----------------
    df_out.write.mode("overwrite").parquet(silver_path)
    print(f"[SILVER][financials] wrote: {silver_path} rows={df_out.count()} cols={len(df_out.columns)}")
