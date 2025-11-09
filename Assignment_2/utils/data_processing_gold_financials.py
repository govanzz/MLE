# utils/data_processing_gold_financials.py
from datetime import datetime
from pyspark.sql import functions as F, types as T
from pyspark.sql import Window

# ---------- helpers ----------
def _token(snapshot_date_str: str) -> str:
    return datetime.strptime(snapshot_date_str, "%Y-%m-%d").strftime("%Y_%m_%d")

def _safe_div(num, den):
    return F.when((den.isNull()) | (den == 0), None).otherwise(num / den)

def _util_bucket(col):
    return (
        F.when(col < 10, "U0:[0,10)")
         .when(col < 30, "U1:[10,30)")
         .when(col < 50, "U2:[30,50)")
         .when(col < 80, "U3:[50,80)")
         .otherwise("U4:[80,+)")
    )

def _util_bucket_idx(col):
    return (
        F.when(col == "U0:[0,10)", 0)
         .when(col == "U1:[10,30)", 1)
         .when(col == "U2:[30,50)", 2)
         .when(col == "U3:[50,80)", 3)
         .when(col == "U4:[80,+)", 4)
    )

def _hist_bucket(col):
    return (
        F.when(col < 12, "H0:<1y")
         .when(col < 36, "H1:1-3y")
         .when(col < 60, "H2:3-5y")
         .when(col < 120, "H3:5-10y")
         .otherwise("H4:>=10y")
    )

def _hist_bucket_idx(col):
    return (
        F.when(col == "H0:<1y", 0)
         .when(col == "H1:1-3y", 1)
         .when(col == "H2:3-5y", 2)
         .when(col == "H3:5-10y", 3)
         .when(col == "H4:>=10y", 4)
    )

def _inq_bucket(col):
    return (
        F.when(col.isNull(), "I0:NA")
         .when(col == 0, "I1:0")
         .when(col <= 2, "I2:1-2")
         .when(col <= 5, "I3:3-5")
         .otherwise("I4:>5")
    )

def _inq_bucket_idx(col):
    return (
        F.when(col == "I0:NA", 0)
         .when(col == "I1:0", 1)
         .when(col == "I2:1-2", 2)
         .when(col == "I3:3-5", 3)
         .when(col == "I4:>5", 4)
    )

def _normalize_loan_types(col):
    # Uppercase, replace "AND"→",", remove "NOT SPECIFIED", tidy commas/spaces
    up = F.upper(F.coalesce(col.cast("string"), F.lit("")))
    s1 = F.regexp_replace(up, r"\bAND\b", ",")
    s2 = F.regexp_replace(s1, r"NOT SPECIFIED", "")
    s3 = F.regexp_replace(s2, r"\s+", " ")
    s4 = F.regexp_replace(s3, r",\s*,", ",")
    s5 = F.regexp_replace(s4, r"(^\s*,)|(,\s*$)", "")
    return s5

# ---------- main processor ----------
def process_gold_financials(
    snapshot_date_str: str,
    silver_financials_directory: str,
    gold_financials_features_dir: str,
    gold_financials_wide_dir: str,
    spark,
):
    """
    Reads:  {silver_financials_directory}/silver_financials_<YYYY_MM_DD>.parquet
    Writes: {gold_financials_features_dir}/gold_financials_features_<YYYY_MM_DD>.parquet
            {gold_financials_wide_dir}/gold_financials_wide_<YYYY_MM_DD>.parquet

    Spec: no winsorization, exclude delinquency fields from modeling,
          keep base numerics only in _wide (not in _features).
    """
    tok = _token(snapshot_date_str)
    in_path  = f"{silver_financials_directory}/silver_financials_{tok}.parquet"
    out_feat = f"{gold_financials_features_dir}/gold_financials_features_{tok}.parquet"
    out_wide = f"{gold_financials_wide_dir}/gold_financials_wide_{tok}.parquet"

    df = spark.read.parquet(in_path)
    # Drop delinquency fields entirely (unclear timing)
    df = df.drop("delay_from_due_date", "num_of_delayed_payment")

    # ---------- missingness flags ----------
    d = (df
         .withColumn("is_income_null",               F.col("annual_income").isNull().cast("int"))
         .withColumn("is_outstanding_debt_null",     F.col("outstanding_debt").isNull().cast("int"))
         .withColumn("is_util_null",                 F.col("credit_utilization_ratio").isNull().cast("int"))
         .withColumn("is_hist_null",                 F.col("credit_history_age").isNull().cast("int"))
         .withColumn("is_inquiry_null",              F.col("num_credit_inquiries").isNull().cast("int"))
         .withColumn("is_behav_null",                F.col("payment_behaviour").isNull().cast("int"))
         .withColumn("is_type_of_loan_null",         F.col("type_of_loan").isNull().cast("int"))
    )
    # payment_of_min_amount: "NM" → unknown → null flag
    up_min = F.upper(F.col("payment_of_min_amount"))
    d = d.withColumn(
        "is_minpay_null",
        (up_min.isNull() | (up_min.isin("NM", "NULL", ""))).cast("int")
    )

    # ---------- affordability ratios (safe divide) ----------
    d = (d
         .withColumn("dti",              _safe_div(F.col("outstanding_debt"), F.col("annual_income")))
         .withColumn("emi_to_income",    _safe_div(F.col("total_emi_per_month") * F.lit(12.0), F.col("annual_income")))
         .withColumn("emi_to_balance",   _safe_div(F.col("total_emi_per_month"), F.abs(F.col("monthly_balance"))))
         .withColumn("invest_to_income", _safe_div(F.col("amount_invested_monthly") * F.lit(12.0), F.col("annual_income")))
    )

    # ---------- utilization ----------
    util = F.col("credit_utilization_ratio").cast("double")
    d = (d
         .withColumn("util_capped", F.when(util > 150, 150.0).otherwise(util))
         .withColumn("util_sq", F.pow(F.col("util_capped"), F.lit(2.0)))
    )
    d = d.withColumn("util_bucket", _util_bucket(F.col("credit_utilization_ratio")))
    d = d.withColumn("util_bucket_idx", _util_bucket_idx(F.col("util_bucket")))
    d = d.withColumn("emi_x_util", F.col("total_emi_per_month") * F.col("credit_utilization_ratio"))

    # ---------- interest & pricing ----------
    d = d.withColumn("interest_x_loans", F.col("interest_rate") * F.col("num_of_loan"))

    # ---------- history & inquiries ----------
    d = d.withColumn("hist_bucket", _hist_bucket(F.col("credit_history_age")))
    d = d.withColumn("hist_bucket_idx", _hist_bucket_idx(F.col("hist_bucket")))
    d = d.withColumn("inq_bucket", _inq_bucket(F.col("num_credit_inquiries")))
    d = d.withColumn("inq_bucket_idx", _inq_bucket_idx(F.col("inq_bucket")))
    d = d.withColumn("dti_x_inq", F.col("dti") * F.col("num_credit_inquiries"))

    # ---------- payment behaviour & policy ----------
    d = d.withColumn(
        "minpay_flag",
        F.when(F.upper(F.col("payment_of_min_amount")) == "YES", 1)
         .when(F.upper(F.col("payment_of_min_amount")) == "NO", 0)
         .otherwise(F.lit(None).cast("int"))
    )
    beh_up = F.upper(F.coalesce(F.col("payment_behaviour"), F.lit("")))
    d = d.withColumn(
        "pay_behav_coarse",
        F.when(beh_up.contains("LATE") | beh_up.contains("DELAY"), "late")
         .when(beh_up.contains("MIN") | beh_up.contains("PART") | beh_up.contains("IRREG"), "irregular")
         .when(beh_up.contains("REG") | beh_up.contains("ON TIME"), "regular")
         .otherwise("unknown")
    )

    # ---------- loan-mix from type_of_loan ----------
    tol_norm = _normalize_loan_types(F.col("type_of_loan"))
    parts = F.split(tol_norm, r"\s*,\s*")
    parts_dedup = F.array_distinct(F.filter(parts, lambda x: x.isNotNull() & (x != "")))
    d = d.withColumn("loan_types_count_unique", F.size(parts_dedup))
    # presence flags
    d = (d
         .withColumn("has_HOME",      F.array_contains(parts_dedup, F.lit("HOME LOAN")))
         .withColumn("has_AUTO",      F.array_contains(parts_dedup, F.lit("AUTO LOAN")))
         .withColumn("has_CC",        F.array_contains(parts_dedup, F.lit("CREDIT CARD")))
         .withColumn("has_PERSONAL",  F.array_contains(parts_dedup, F.lit("PERSONAL LOAN")))
         .withColumn("has_STUDENT",   F.array_contains(parts_dedup, F.lit("STUDENT LOAN")))
         .withColumn("has_MORTGAGE",  F.array_contains(parts_dedup, F.lit("MORTGAGE LOAN")))
    )

    # ---------- select outputs ----------
    keys = ["customer_id", "snapshot_date", "snap_ym"]
    pass_through = ["interest_rate", "num_credit_inquiries", "credit_mix", "num_of_loan",
                    "credit_history_age", "credit_utilization_ratio"]
    ratios = ["dti", "emi_to_income", "emi_to_balance", "invest_to_income"]
    util_cols = ["util_capped", "util_sq", "util_bucket", "util_bucket_idx", "emi_x_util"]
    interest_cols = ["interest_x_loans"]
    hist_inq = ["hist_bucket", "hist_bucket_idx", "inq_bucket", "inq_bucket_idx", "dti_x_inq"]
    pay_cols = ["minpay_flag", "pay_behav_coarse"]
    loanmix = ["loan_types_count_unique", "has_HOME", "has_AUTO", "has_CC", "has_PERSONAL", "has_STUDENT", "has_MORTGAGE"]
    miss = ["is_income_null", "is_outstanding_debt_null", "is_util_null", "is_hist_null",
            "is_inquiry_null", "is_minpay_null", "is_behav_null", "is_type_of_loan_null"]

    feature_cols = keys + pass_through + ratios + util_cols + interest_cols + hist_inq + pay_cols + loanmix + miss

    # Gold FEATURES: exclude base numerics used in ratios
    d.select(*feature_cols).write.mode("overwrite").parquet(out_feat)

    # Gold WIDE (audit): Silver + engineered (still no delinquency fields)
    engineered_cols = list(set(feature_cols) - set(keys) - set(["credit_history_age", "credit_utilization_ratio",
                                                                "interest_rate", "num_credit_inquiries",
                                                                "credit_mix", "num_of_loan"]))
    wide = df.join(d.select(keys + engineered_cols), keys, "left")
    wide.write.mode("overwrite").parquet(out_wide)

    print(f"[GOLD][financials_features] wrote: {out_feat}")
    print(f"[GOLD][financials_wide]     wrote: {out_wide}")
