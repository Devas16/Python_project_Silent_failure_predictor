"""
app/api.py
----------
FastAPI REST API for the Silent Failure Predictor.

Endpoints:
    GET  /health                — service health check
    POST /predict               — score a batch of metric readings
    POST /predict/single        — score a single metric reading
    GET  /model/info            — model metadata
    GET  /metrics/sample        — return a sample row from the dataset

Run:
    uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload

Example request:
    curl -X POST http://localhost:8000/predict/single \
         -H "Content-Type: application/json" \
         -d '{"cpu_usage": 95, "memory_usage": 88, "response_time_ms": 2500,
              "disk_io_mbps": 180, "network_latency_ms": 250, "error_rate_pct": 12}'
"""

import os
import time
import numpy as np
import pandas as pd
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config_loader import load_config
from src.logger import log

# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------
cfg = load_config()

app = FastAPI(
    title       = "Silent Failure Predictor API",
    description = "REST API for ML-based anomaly detection in system metrics",
    version     = "1.0.0",
)

# Allow all origins in dev — restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ---------------------------------------------------------------------------
# Lazy model loading — models are loaded once on first request
# ---------------------------------------------------------------------------
_detector    = None
_preprocessor = None


def _load_models():
    global _detector, _preprocessor
    if _detector is not None:
        return

    from src.isolation_forest_model import IsolationForestDetector
    from src.preprocessor import Preprocessor

    model_path  = cfg["isolation_forest"]["model_path"]
    scaler_path = "models/scaler.pkl"

    if not os.path.exists(model_path):
        raise RuntimeError(
            f"Model not found at '{model_path}'. "
            "Run `python -m src.train` first."
        )

    _detector = IsolationForestDetector()
    _detector.load(model_path)

    _preprocessor = Preprocessor()
    _preprocessor.load(scaler_path)

    log.info("Models loaded for API serving.")


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class MetricReading(BaseModel):
    """A single row of system metrics."""
    cpu_usage:          float = Field(..., ge=0,   le=100,  description="CPU usage (%)")
    memory_usage:       float = Field(..., ge=0,   le=100,  description="Memory usage (%)")
    response_time_ms:   float = Field(..., ge=0,           description="Response time (ms)")
    disk_io_mbps:       float = Field(..., ge=0,           description="Disk I/O (MB/s)")
    network_latency_ms: float = Field(..., ge=0,           description="Network latency (ms)")
    error_rate_pct:     float = Field(..., ge=0,   le=100, description="Error rate (%)")


class BatchReadings(BaseModel):
    """A list of metric readings for batch prediction."""
    readings: List[MetricReading]


class PredictionResult(BaseModel):
    """Prediction response for a single reading."""
    is_anomaly:    bool
    anomaly_score: float
    severity:      str   # "normal" | "warning" | "critical"
    message:       str


class BatchPredictionResult(BaseModel):
    total_samples:    int
    anomaly_count:    int
    anomaly_fraction: float
    predictions:      List[PredictionResult]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _score_to_severity(score: float, threshold_warn: float = 0.3,
                        threshold_crit: float = 0.5) -> str:
    if score < threshold_warn:
        return "normal"
    elif score < threshold_crit:
        return "warning"
    return "critical"


def _readings_to_df(readings: List[MetricReading]) -> pd.DataFrame:
    return pd.DataFrame([r.model_dump() for r in readings])


def _predict(df: pd.DataFrame):
    _load_models()
    X      = _preprocessor.transform(df)
    labels, scores = _detector.predict(X)
    return labels, scores


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health():
    """Liveness probe."""
    return {"status": "ok", "timestamp": time.time()}


@app.get("/model/info", tags=["System"])
def model_info():
    """Return metadata about the loaded model."""
    return {
        "model_type":      "Isolation Forest",
        "contamination":   cfg["isolation_forest"]["contamination"],
        "n_estimators":    cfg["isolation_forest"]["n_estimators"],
        "features":        cfg["features"]["features_list"],
        "rolling_window":  cfg["features"]["rolling_window"],
    }


@app.post("/predict/single", response_model=PredictionResult, tags=["Prediction"])
def predict_single(reading: MetricReading):
    """
    Score a single metric reading.

    Returns whether the reading is an anomaly along with its anomaly score
    and a human-readable severity label.
    """
    try:
        df            = _readings_to_df([reading])
        labels, scores = _predict(df)
        is_anom       = bool(labels[0])
        score         = float(scores[0])
        severity      = _score_to_severity(score)

        return PredictionResult(
            is_anomaly    = is_anom,
            anomaly_score = round(score, 6),
            severity      = severity,
            message       = (
                "⚠️  Silent failure detected — immediate investigation recommended."
                if is_anom else
                "✅  System operating normally."
            ),
        )
    except Exception as exc:
        log.error(f"predict_single error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict", response_model=BatchPredictionResult, tags=["Prediction"])
def predict_batch(payload: BatchReadings):
    """
    Score a batch of metric readings.

    Returns per-sample predictions and aggregate anomaly statistics.
    """
    try:
        df            = _readings_to_df(payload.readings)
        labels, scores = _predict(df)

        results = [
            PredictionResult(
                is_anomaly    = bool(l),
                anomaly_score = round(float(s), 6),
                severity      = _score_to_severity(float(s)),
                message       = (
                    "⚠️  Silent failure detected."
                    if l else "✅  Normal."
                ),
            )
            for l, s in zip(labels, scores)
        ]

        return BatchPredictionResult(
            total_samples    = len(labels),
            anomaly_count    = int(labels.sum()),
            anomaly_fraction = round(float(labels.mean()), 4),
            predictions      = results,
        )
    except Exception as exc:
        log.error(f"predict_batch error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/metrics/sample", tags=["Data"])
def sample_metrics():
    """Return a random row from the stored dataset for quick testing."""
    from src.data_generator import load_dataset
    df  = load_dataset()
    row = df.sample(1).iloc[0]
    return row.to_dict()
