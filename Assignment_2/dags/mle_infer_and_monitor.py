from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator  # Airflow 2.6.x+

PROJECT = "/opt/airflow/project"

def ymd_under(ds_expr: str) -> str:
    # "2024-06-01" -> "2024_06_01"
    return "{{ macros.ds_format(" + ds_expr + ", '%Y-%m-%d', '%Y_%m_%d') }}"

with DAG(
    dag_id="mle_infer_and_monitor",
    description="Score the ds month with latest model, then update monitoring",
    start_date=datetime(2023, 7, 1),   # aligns with first month a model exists
    schedule_interval="@monthly",
    catchup=True,
    max_active_runs=1,
    tags=["mle","a2","infer","monitor"],
) as dag:

    gold_file = "gold_model_dataset_" + ymd_under("ds") + ".parquet"

    infer_current_month = BashOperator(
        task_id="infer_current_month",
        bash_command=(
            f"cd {PROJECT} && "
            "python infer.py "
            "--input datamart/gold/model_dataset/" + gold_file + " "
            "--id-cols customer_id loan_id"
        ),
    )

    monitor_trend = BashOperator(
        task_id="monitor_trend",
        bash_command=(
            f"cd {PROJECT} && "
            "python monitor.py --prefer xgb"
        ),
    )

    infer_current_month >> monitor_trend
