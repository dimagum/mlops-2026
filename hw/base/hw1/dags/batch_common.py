import hashlib
import json
import math
import os
import re
import tempfile
from datetime import date
from pathlib import Path

import numpy as np

FEATURES = ["age", "tenure_months", "monthly_spend", "support_tickets_30d"]
LIMITS = [(18, 100), (0, 240), (0, 10000), (0, 100)]
ARTIFACTS = Path(os.environ.get("BATCH_ARTIFACTS", "/opt/airflow/include/artifacts"))


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode()


def digest(content):
    return hashlib.sha256(content).hexdigest()


def save(value):
    content = canonical(value)
    sha = digest(content)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=ARTIFACTS, delete=False) as file:
        file.write(content)
        temporary = Path(file.name)
    temporary.replace(ARTIFACTS / f"{sha}.json")
    return {"sha256": sha}


def read(ref):
    sha = ref["sha256"]
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise ValueError("Invalid artifact reference")
    content = (ARTIFACTS / f"{sha}.json").read_bytes()
    if digest(content) != sha:
        raise ValueError("Artifact checksum mismatch")
    return json.loads(content)


def business_date(explicit, interval_start):
    if explicit:
        return date.fromisoformat(explicit).isoformat()
    if interval_start is None:
        raise ValueError("business_date is required without a data interval")
    return interval_start.in_timezone("Europe/Moscow").to_date_string()


def validate_rows(rows, day):
    if not rows:
        raise ValueError("Empty rows")

    seen_ids = set()

    for row in rows:
        customer_id = row["customer_id"]

        if not isinstance(customer_id, str) or not customer_id:
            raise ValueError("Invalid customer_id")

        if customer_id in seen_ids:
            raise ValueError(f"Duplicate customer_id: {customer_id}")

        seen_ids.add(customer_id)

        if row["business_date"] != day:
            raise ValueError("Business date mismatch")

        for name, (lower, upper) in zip(FEATURES, LIMITS):
            value = row[name]

            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
            ):
                raise ValueError(f"Invalid feature: {name}")

            if not (lower <= value <= upper):
                raise ValueError(f"Feature out of range: {name}")

    return rows 


def validate_model(model):
    if model["feature_names"] != FEATURES:
        raise ValueError("Feature order mismatch")
    for field in ("mean", "scale", "weights"):
        if len(model[field]) != len(FEATURES):
            raise ValueError("Model dimensions mismatch")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in model[field]):
            raise ValueError("Invalid model coefficient")
    if any(v <= 0 for v in model["scale"]) or type(model["bias"]) not in (int, float) or not math.isfinite(model["bias"]):
        raise ValueError("Invalid model scale or bias")


def predict(rows, model):
    validate_model(model)
    matrix = np.asarray([[row[name] for name in FEATURES] for row in rows], dtype=np.float64)
    normalized = (matrix - np.asarray(model["mean"])) / np.asarray(model["scale"])
    logits = normalized @ np.asarray(model["weights"]) + model["bias"]
    scores = np.round(1 / (1 + np.exp(-np.clip(logits, -700, 700))), 8)
    return [{"customer_id": row["customer_id"], "score": float(score), "label": int(score >= 0.5)}
            for row, score in zip(rows, scores)]


def infer_batches(rows, model, batch_size):
    if (
        type(batch_size) is not int
        or not (1 <= batch_size <= 1000)
    ):
        raise ValueError("Invalid batch_size")

    predictions = []

    for start in range(0, len(rows), batch_size):
        batch = rows[start:min(start + batch_size, len(rows))]

        print(
            f"Batch {start // batch_size + 1}: "
            f"size={len(batch)}"
        )

        batch_predictions = predict(batch, model)

        predictions.extend(batch_predictions)

    return predictions


def validate_predictions(predictions, expected_ids):
    if len(predictions) != len(expected_ids):
        raise ValueError("Predictions count mismatch")

    actual_ids = [
        prediction["customer_id"]
        for prediction in predictions
    ]

    if len(set(actual_ids)) != len(actual_ids):
        raise ValueError("Duplicate predictions")

    if set(actual_ids) != set(expected_ids):
        raise ValueError("Customer ids mismatch")

    for prediction in predictions:
        score = prediction["score"]

        if (
            type(score) not in (int, float)
            or not math.isfinite(score)
            or not (0 <= score <= 1)
        ):
            raise ValueError("Invalid score")

        label = prediction["label"]

        if type(label) is not int or label not in (0, 1):
            raise ValueError("Invalid label")

        if label != int(score >= 0.5):
            raise ValueError("Label does not match score")


def replace_day(conn, day, model_version, predictions):
    if not predictions:
        raise ValueError("Empty predictions")

    with conn.transaction(), conn.cursor() as cur:
        lock_key = day + ":" + model_version

        cur.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
            (lock_key,),
        )

        cur.execute(
            """
            DELETE FROM predictions
            WHERE business_date = %s
              AND model_version = %s
            """,
            (day, model_version),
        )

        for prediction in predictions:
            cur.execute(
                """
                INSERT INTO predictions (
                    business_date,
                    model_version,
                    customer_id,
                    score,
                    label
                )
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    day,
                    model_version,
                    prediction["customer_id"],
                    prediction["score"],
                    prediction["label"],
                ),
            )
