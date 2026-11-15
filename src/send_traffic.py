"""Replay the newest real transactions (the test period the model never trained on) against the API,
then print what was blocked and the drift report.

Usage:  .venv/Scripts/python src/send_traffic.py [API_URL] [N_ROWS]
"""
import sys
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
url = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080").rstrip("/")
n = int(sys.argv[2]) if len(sys.argv) > 2 else 2000

features = httpx.get(f"{url}/health", timeout=60).json()["features"]
df = pd.read_csv(ROOT / "data/raw/train_transaction.csv.zip", usecols=["TransactionDT", "isFraud", *features])
df = df.sort_values("TransactionDT").tail(n)

results = []
for start in range(0, n, 500):
    chunk = df.iloc[start:start + 500]
    records = chunk[features].astype(object).where(chunk[features].notna(), None).to_dict("records")
    results += httpx.post(f"{url}/predict", json={"transactions": records}, timeout=60).json()["results"]

blocked = pd.Series([r["block"] for r in results], index=df.index)
print(f"sent {n} | blocked {blocked.mean():.1%} | fraud caught {blocked[df.isFraud == 1].mean():.1%} "
      f"of {int(df.isFraud.sum())} | innocent blocked {blocked[df.isFraud == 0].mean():.1%}")
report = httpx.get(f"{url}/monitoring", timeout=60).json()
print("drifted features (PSI > 0.2):", report["drifted"])
print("top PSI:", dict(list(report["psi"].items())[:5]))
