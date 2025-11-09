from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import ShortCircuitOperator
from airflow.models import Variable

PROJECT = "/opt/airflow/project"

# ---- Retrain SOP (governance) ----------------------------------------
# Set in Airflow UI → Admin → Variables → Key: mle_retrain_mode
# Values:
#   "always"       -> retrain every month (default)
#   "manual_skip"  -> skip training (use when freezing model)
RETRAIN_MODE = Variable.get("mle_retrain_mode", default_var="always")

def should_retrain() -> bool:
    return RETRAIN_MODE != "manual_skip"

# ---- Date helpers (relative to ds = first-of-month) -------------------
def ds_add(days: int) -> str:
    return "{{ macros.ds_add(ds, " + str(days) + ") }}"
def month_first(ds_expr: str) -> str:
    return "{{ macros.ds_format(" + ds_expr + ", '%Y-%m-%d', '%Y-%m-01') }}"

with DAG(
    dag_id="mle_training_rollup",
    description="Monthly training: rolling 2-1-1 + OOT (keeps Train ≥ 2023-07)",
    start_date=datetime(2023, 11, 1),     # earliest ds so Train==Jul–Aug 2023
    schedule_interval="@monthly",
    catchup=True,
    max_active_runs=1,
    tags=["mle","a2","train"],
) as dag:

    # Windows for a run on ds = Month M (1st of month):
    # Train: M-4..M-3, Valid: M-2, Test: M-1, OOT: M
    train_start = month_first(ds_add(-120))  # ~M-4
    train_end   = month_first(ds_add(-90))   # ~M-3
    valid_start = month_first(ds_add(-60))   # ~M-2
    valid_end   = month_first(ds_add(-60))
    test_start  = month_first(ds_add(-30))   # ~M-1
    test_end    = month_first(ds_add(-30))
    oot_start   = month_first("ds")          # M
    oot_end     = month_first("ds")

    gate_retrain = ShortCircuitOperator(
        task_id="gate_retrain",
        python_callable=should_retrain,
    )

    train_models = BashOperator(
        task_id="train_models_2_1_1",
    bash_command=(
        f"cd {PROJECT} && "
        "python train.py "
        "--train-start {{ macros.ds_format(macros.ds_add(ds, -120), '%Y-%m-%d', '%Y-%m-01') }} "
        "--train-end   {{ macros.ds_format(macros.ds_add(ds, -90),  '%Y-%m-%d', '%Y-%m-01') }} "
        "--valid-start {{ macros.ds_format(macros.ds_add(ds, -60),  '%Y-%m-%d', '%Y-%m-01') }} "
        "--valid-end   {{ macros.ds_format(macros.ds_add(ds, -60),  '%Y-%m-%d', '%Y-%m-01') }} "
        "--test-start  {{ macros.ds_format(macros.ds_add(ds, -30),  '%Y-%m-%d', '%Y-%m-01') }} "
        "--test-end    {{ macros.ds_format(macros.ds_add(ds, -30),  '%Y-%m-%d', '%Y-%m-01') }} "
        "--oot-start   {{ macros.ds_format(ds, '%Y-%m-%d', '%Y-%m-01') }} "
        "--oot-end     {{ macros.ds_format(ds, '%Y-%m-%d', '%Y-%m-01') }}"
    ),
    )

    gate_retrain >> train_models
