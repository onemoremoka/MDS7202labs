from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime
import os

from hiring_functions import create_folders, split_data, preprocess_and_train, gradio_interface

# Definimos los argumentos por defecto
default_args = {
    'owner': 'airflow',
}

# Creamos el DAG
with DAG(
    dag_id='hiring_lineal', # identificador del DAG
    default_args=default_args,
    description='Pipeline lineal para predicción de contrataciones',
    schedule=None,  # ejecución manual
    start_date=datetime(2024, 10, 1),
    catchup=False, # sin backfill
    tags=['hiring', 'random_forest']
) as dag:

    start = EmptyOperator(
        task_id='start'
        )

    task_create_folders = PythonOperator(
        task_id='create_folders',
        python_callable=create_folders,
        op_kwargs={'ds': '{{ ds }}'},
    )

    def download_data(ds):
        output_path = f"dags/output/{ds}/raw/data_1.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        os.system(f"curl -o {output_path} https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv")

    task_download_data = PythonOperator(
        task_id='download_data',
        python_callable=download_data,
        op_kwargs={'ds': '{{ ds }}'},
    )

    task_split_data = PythonOperator(
        task_id='split_data',
        python_callable=split_data,
        op_kwargs={'ds': '{{ ds }}'},
    )

    task_preprocess_and_train = PythonOperator(
        task_id='preprocess_and_train',
        python_callable=  preprocess_and_train,
        op_kwargs={'ds': '{{ ds }}'},
    )

    gradio_task = PythonOperator(
        task_id='gradio_interface',
        python_callable=gradio_interface,
        op_kwargs={'ds': '{{ ds }}'},
    )

    end = EmptyOperator(
        task_id='end'
    )

    start >> task_create_folders >> task_download_data >> task_split_data >> task_preprocess_and_train >> gradio_task >> end
