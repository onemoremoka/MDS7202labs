from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from hiring_dynamic_functions import (  # Cambia esto según dónde guardes las funciones
    create_folders, download_data, load_and_merge, split_data, train_model, evaluate_models
)

def branch_download(**context):
    """Decide qué archivos descargar según la fecha de ejecución."""
    ds = context['ds']
    fecha = datetime.strptime(ds, '%Y-%m-%d')
    if fecha < datetime(2024, 11, 1):
        return "download_data1"
    else:
        return "download_data2"

with DAG(
    dag_id='dynamic_hiring_pipeline',
    schedule_interval='0 15 5 * *',  # Día 5 de cada mes, 15:00 UTC
    start_date=datetime(2024, 10, 1),
    catchup=True,
    tags=["dynamic", "hiring"],
    default_args={"retries": 1, "retry_delay": timedelta(minutes=2)},
) as dag:
    
    start = EmptyOperator(task_id="start")

    create_folders_task = PythonOperator(
        task_id="create_folders",
        python_callable=create_folders,
        op_kwargs={"ds": "{{ ds }}"},
    )

    branch_download_task = BranchPythonOperator(
        task_id='branch_download',
        python_callable=branch_download,
        provide_context=True,
    )
    download_data1 = PythonOperator(
        task_id="download_data1",
        python_callable=download_data,
        op_kwargs={"ds": "{{ ds }}", "download_data2": False},
    )
    download_data2 = PythonOperator(
        task_id="download_data2",
        python_callable=download_data,
        op_kwargs={"ds": "{{ ds }}", "download_data2": True},
    )
    join = EmptyOperator(task_id="join", trigger_rule="none_failed_min_one_success")

    load_and_merge_task = PythonOperator(
        task_id="load_and_merge",
        python_callable=load_and_merge,
        op_kwargs={"ds": "{{ ds }}"},
    )
    split_data_task = PythonOperator(
        task_id="split_data",
        python_callable=split_data,
        op_kwargs={"ds": "{{ ds }}"},
    )

    train_rf = PythonOperator(
        task_id="train_random_forest",
        python_callable=train_model,
        op_kwargs={
            "ds": "{{ ds }}",
            "model_class": RandomForestClassifier,
            "model_name": "random_forest",
        },
    )
    train_logreg = PythonOperator(
        task_id="train_logistic_regression",
        python_callable=train_model,
        op_kwargs={
            "ds": "{{ ds }}",
            "model_class": LogisticRegression,
            "model_name": "logistic_regression",
        },
    )
    train_tree = PythonOperator(
        task_id="train_decision_tree",
        python_callable=train_model,
        op_kwargs={
            "ds": "{{ ds }}",
            "model_class": DecisionTreeClassifier,
            "model_name": "decision_tree",
        },
    )

    evaluate_task = PythonOperator(
        task_id="evaluate_models",
        python_callable=evaluate_models,
        op_kwargs={"ds": "{{ ds }}"},
        trigger_rule="all_success",
    )
    end = EmptyOperator(task_id="end")

    # Encadenamiento
    start >> create_folders_task >> branch_download_task
    branch_download_task >> [download_data1, download_data2] >> join
    join >> load_and_merge_task >> split_data_task
    split_data_task >> [train_rf, train_logreg, train_tree] >> evaluate_task >> end
