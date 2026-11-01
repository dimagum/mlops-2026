# TODO: объявите здесь Airflow DAG hw_basic_batch_inference (extract - infer - publish).
# См. раздел "Задание 1" в ASSIGNMENT.md.

from datetime import timedelta
import requests

import pendulum
from airflow.sdk import Param, dag, task, BaseHook
from airflow.providers.postgres.hooks.postgres import PostgresHook

from batch_common import (
    business_date, 
    digest, 
    save, 
    read, 
    replace_day,
    validate_predictions,
    validate_rows,
    infer_batches
)


@task
def extract(**context):
    params = context["params"]

    explicit_day = params["business_date"]
    interval_start = context.get("data_interval_start")



    day = business_date(explicit_day, interval_start)

    connection = BaseHook.get_connection(conn_id="source_api")

    host = connection.host.rstrip("/")

    rows_response = requests.get(
        f"{host}/basic/{day}.json",
        timeout=10,
    )
    rows_response.raise_for_status()
    payload = rows_response.json()

    if payload["business_date"] != day:
        raise ValueError(f"Expected business_date {day}, got {payload['business_date']}")

    model_response = requests.get(f"{host}/models/v1.json", timeout=10)

    model_response.raise_for_status()

    model_sha256 = digest(model_response.content)

    return save({
        "business_date": day,
        "rows": payload["rows"],
        "model": model_response.json(),
        "model_sha256": model_sha256,
    })


@task(retries=0)
def infer(extract_ref, **context):
    payload = read(extract_ref) 

    rows = validate_rows(
        payload["rows"],
        payload["business_date"],
    )

    batch_size = context["params"]["batch_size"]

    predictions = infer_batches(rows, payload["model"], batch_size,) 

    expected_ids = [row["customer_id"] for row in rows]

    validate_predictions(predictions, expected_ids,)

    return save({
        "business_date": payload["business_date"],
        "model_version": "v1",
        "predictions": predictions,
        "model_sha256": payload["model_sha256"],
    })



@task
def publish(infer_ref):
    payload = read(infer_ref)

    conn = PostgresHook(
        postgres_conn_id="warehouse",
    ).get_conn()

    try:
        replace_day(
            conn,
            payload["business_date"],
            payload["model_version"],
            payload["predictions"],
        )

        return {
            "business_date": payload["business_date"],
            "model_version": payload["model_version"],
            "rows": len(payload["predictions"]),
        }

    finally:
        conn.close()


@dag(
    dag_id="hw_basic_batch_inference",
    start_date=pendulum.datetime(2026, 9, 1, tz="Europe/Moscow"),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["homework", "basic"],
    params={
        "business_date": Param(
            default="2026-09-01",
            type="string",
            format="date",
        ),
        "batch_size": Param(
            default=32,
            type="integer",
            minimum=1,
            maximum=1000,
        ),
    },
    default_args={
        "retries": 2,
        "retry_delay": timedelta(seconds=5),
        "execution_timeout": timedelta(minutes=2),
    },
)
def hw_basic_batch_inference():
    extract_ref = extract()
    infer_ref = infer(extract_ref)
    publish(infer_ref)

hw_basic_batch_inference()
