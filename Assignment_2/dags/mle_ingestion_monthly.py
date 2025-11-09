from datetime import datetime
from airflow import DAG
from airflow.operators.bash_operator import BashOperator  # Airflow 2.6.x compatible import

PROJECT = "/opt/airflow/project"

with DAG(
    dag_id="mle_ingestion_monthly",
    description="Build Gold model_dataset for each month (creates datamart if missing)",
    start_date=datetime(2023, 1, 1),   # your earliest month
    schedule_interval="@monthly",
    catchup=True,
    max_active_runs=1,
    tags=["mle","a2","ingestion"],
) as dag:

    build_gold_for_month = BashOperator(
        task_id="build_gold_for_month",
        bash_command=(
            f"cd {PROJECT} && "
            "python datapipeline.py --start {{ ds }} --end {{ ds }}"
        ),
    )
