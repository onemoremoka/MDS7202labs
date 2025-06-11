FROM python:3.10-slim

WORKDIR /root/airflow
ENV AIRFLOW_HOME=/root/airflow

RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install --no-cache-dir 'pendulum==2.1.2'
RUN pip install --no-cache-dir scikit-learn
RUN pip install --no-cache-dir pandas
RUN pip install --no-cache-dir numpy
RUN pip install --no-cache-dir lightgbm
RUN pip install --no-cache-dir joblib
RUN pip install --no-cache-dir apache-airflow
RUN pip install --no-cache-dir Flask-Session
RUN pip install --no-cache-dir gradio

COPY ./dags $AIRFLOW_HOME/dags

EXPOSE 8080
