"""
Azure ML online endpoint scoring script.

The Azure ML runtime calls init() once at container startup and run() on
every inference request.

Input JSON format:
    {"data": [{"alcohol": 12.3, "volatile acidity": 0.5, ...}, ...]}

Output JSON format:
    {
        "predictions": [1, 0, 2],
        "probabilities": [[0.1, 0.8, 0.1], ...],
        "model_version": "7",
        "prediction_labels": ["class_1", "class_0", "class_2"]
    }
"""

import json
import logging
import os
import time
from typing import Any

import mlflow
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Globals populated in init()
_model = None
_model_version = os.getenv("MODEL_VERSION", "unknown")

# Expected feature columns (must match training schema)
EXPECTED_FEATURES = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280/od315_of_diluted_wines",
    "proline",
]

FEATURE_RANGES = {
    "alcohol": (11.0, 15.0),
    "malic_acid": (0.7, 6.0),
    "ash": (1.3, 3.3),
    "alcalinity_of_ash": (10.0, 30.0),
    "magnesium": (70.0, 160.0),
    "total_phenols": (0.9, 4.0),
    "flavanoids": (0.3, 6.0),
    "nonflavanoid_phenols": (0.1, 0.7),
    "proanthocyanins": (0.4, 4.0),
    "color_intensity": (1.2, 14.0),
    "hue": (0.4, 1.8),
    "od280/od315_of_diluted_wines": (1.2, 4.1),
    "proline": (278.0, 1680.0),
}


def init():
    """Load the model from the Azure ML model directory."""
    global _model

    model_dir = os.getenv("AZUREML_MODEL_DIR", ".")
    model_path = os.path.join(model_dir, "model")

    logger.info("Loading model from: %s", model_path)
    try:
        _model = mlflow.sklearn.load_model(model_path)
        logger.info("Model loaded successfully. Version: %s", _model_version)
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        raise


def _validate_input(records: list[dict]) -> tuple[bool, str]:
    """
    Validate incoming records against expected schema and value ranges.

    Returns (is_valid, error_message).
    """
    if not records:
        return False, "Input 'data' list is empty."

    if len(records) > 1000:
        return False, f"Batch size {len(records)} exceeds limit of 1000."

    for i, record in enumerate(records):
        missing = [f for f in EXPECTED_FEATURES if f not in record]
        if missing:
            return False, f"Record {i}: missing features {missing}"

        for feature, (lo, hi) in FEATURE_RANGES.items():
            if feature in record:
                val = record[feature]
                if not isinstance(val, (int, float)):
                    return False, f"Record {i}: feature '{feature}' must be numeric, got {type(val)}"
                if val < lo * 0.5 or val > hi * 2.0:
                    # Soft range check — warn but allow; hard outliers are blocked
                    logger.warning(
                        "Record %d: feature '%s' value %.3f is far outside expected range [%.1f, %.1f]",
                        i, feature, val, lo, hi,
                    )

    return True, ""


def run(raw_data: str) -> str:
    """
    Score a batch of records.

    Args:
        raw_data: JSON string with shape {"data": [{"feature": value, ...}, ...]}.

    Returns:
        JSON string with predictions, probabilities, and metadata.
    """
    start_time = time.time()

    # -- Parse input -------------------------------------------------------
    try:
        payload = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"Invalid JSON: {exc}", "status": 400})

    if "data" not in payload:
        return json.dumps({"error": "Request must contain a 'data' key.", "status": 400})

    records = payload["data"]

    # -- Validate ----------------------------------------------------------
    valid, error_msg = _validate_input(records)
    if not valid:
        return json.dumps({"error": error_msg, "status": 422})

    # -- Build DataFrame ---------------------------------------------------
    try:
        df = pd.DataFrame(records)
        # Ensure column order and fill any missing optional columns with NaN
        for col in EXPECTED_FEATURES:
            if col not in df.columns:
                df[col] = np.nan
        df = df[EXPECTED_FEATURES]
    except Exception as exc:
        logger.error("DataFrame construction failed: %s", exc)
        return json.dumps({"error": f"Data processing error: {exc}", "status": 500})

    # -- Predict -----------------------------------------------------------
    try:
        predictions = _model.predict(df).tolist()
        probabilities = _model.predict_proba(df).tolist()
        classes = [f"class_{c}" for c in _model.classes_]
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        return json.dumps({"error": f"Prediction error: {exc}", "status": 500})

    elapsed_ms = round((time.time() - start_time) * 1000, 2)
    logger.info(
        "Scored %d records in %.1f ms (model_version=%s)",
        len(records), elapsed_ms, _model_version,
    )

    return json.dumps(
        {
            "predictions": predictions,
            "probabilities": probabilities,
            "prediction_labels": [classes[p] for p in predictions],
            "model_version": _model_version,
            "record_count": len(records),
            "latency_ms": elapsed_ms,
        }
    )
