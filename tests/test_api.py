"""API contract tests against a tiny model trained on synthetic data (no Kaggle data needed, so CI can run them)."""
import importlib
import json
from pathlib import Path

import mlflow.pyfunc
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from xgboost import XGBClassifier

from monitor import make_reference
from scorer import FraudScorer

SRC = Path(__file__).resolve().parents[1] / "src"
FEATURES = ["TransactionAmt", "card1"]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(500, 2)), columns=FEATURES)
    y = (X["TransactionAmt"] > 1).astype(int)
    scorer = FraudScorer("xgboost", XGBClassifier(n_estimators=20).fit(X, y), FEATURES)
    scorer.block_threshold = 0.5
    scorer.reference = {c: make_reference(X[c]) for c in FEATURES} | {"score": make_reference(scorer.score(X))}

    tmp = tmp_path_factory.mktemp("serve")
    model_dir = tmp / "model"
    mlflow.pyfunc.save_model(str(model_dir), python_model=scorer, code_paths=[str(SRC / "scorer.py")])
    (model_dir / "version.json").write_text(json.dumps({"name": "test", "version": "7"}))

    mp = pytest.MonkeyPatch()
    mp.setenv("MODEL_DIR", str(model_dir))
    mp.setenv("REQUEST_LOG", str(tmp / "requests.jsonl"))
    import serve
    yield TestClient(importlib.reload(serve).app)
    mp.undo()


def test_health_reports_model_version(client):
    body = client.get("/health").json()
    assert body["model"]["version"] == "7" and body["features"] == FEATURES


def test_predict_scores_and_blocks(client):
    r = client.post("/predict", json={"transactions": [
        {"TransactionAmt": 3.0, "card1": 0.0},       # clearly "fraud" in the synthetic rule
        {"TransactionAmt": -2.0, "card1": None},     # missing value allowed
        {"TransactionAmt": -2.0, "unknown_field": 1},  # unknown field ignored, missing column allowed
    ]})
    assert r.status_code == 200
    res = r.json()["results"]
    assert all(0 <= x["score"] <= 1 for x in res)
    assert res[0]["block"] and not res[1]["block"]


@pytest.mark.parametrize("payload", [{"transactions": []}, {"transactions": [{"TransactionAmt": "abc"}]}, {}])
def test_bad_input_rejected(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_monitoring_counts_logged_requests(client):
    body = client.get("/monitoring").json()
    assert body["n_requests"] == 3 and "score" in body["psi"]
