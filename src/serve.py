"""Fraud scoring API. Serves the champion exported to model/, logs every request, and reports drift.

Local:  .venv/Scripts/python -m uvicorn serve:app --app-dir src --port 8080
"""
import json
import os
import threading
import time
from pathlib import Path

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from monitor import drift_report

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "model"))
# shortcut: log lives on the container's local disk, so it resets on restart and is per instance;
# move to Cloud Storage/BigQuery when monitoring has to survive deploys.
REQUEST_LOG = Path(os.environ.get("REQUEST_LOG", ROOT / "logs/requests.jsonl"))

scorer = mlflow.pyfunc.load_model(str(MODEL_DIR)).unwrap_python_model()
model_info = json.loads((MODEL_DIR / "version.json").read_text())
_log_lock = threading.Lock()

app = FastAPI(title="Fraud scoring API", version=str(model_info["version"]),
              description="Scores card transactions for fraud. Unknown fields are ignored; missing fields are allowed.")


class Batch(BaseModel):
    # Each transaction: feature name -> number or null. See GET /health for the feature list.
    transactions: list[dict[str, float | None]] = Field(min_length=1, max_length=1000)


@app.get("/health")
def health():
    return {"status": "ok", "model": model_info, "features": scorer.features}


@app.post("/predict")
def predict(batch: Batch):
    df = pd.DataFrame(batch.transactions).reindex(columns=scorer.features).astype(float)
    scores = scorer.score(df)
    REQUEST_LOG.parent.mkdir(parents=True, exist_ok=True)
    with _log_lock:  # requests run in a thread pool; keep appended lines from interleaving
        df.assign(score=scores, ts=time.time(), model_version=model_info["version"]).to_json(
            REQUEST_LOG, orient="records", lines=True, mode="a")
    return {
        "model_version": model_info["version"],
        "results": [{"score": round(float(s), 6), "block": bool(s >= scorer.block_threshold)} for s in scores],
    }


@app.get("/monitoring")
def monitoring():
    """PSI of each feature and of the score: live requests vs the training data."""
    if not REQUEST_LOG.exists():
        return {"n_requests": 0, "psi": {}, "drifted": []}
    with _log_lock:
        current = pd.read_json(REQUEST_LOG, lines=True)
    return {"model_version": model_info["version"], **drift_report(scorer.reference, current)}
