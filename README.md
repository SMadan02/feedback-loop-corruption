# Feedback-Loop Corruption in Fraud Detection

A fraud model blocks risky transactions, but blocked transactions never get a label: nobody
finds out if they were really fraud. Each retrain then learns only from what the model let
through. This project measures that damage and tests a fix.

**Live API:** https://fraud-api-282493443917.europe-west1.run.app/docs

## Status

| Version | What | State |
|---|---|---|
| v0.1 | Baseline fraud model, deployed with versioning and monitoring | Live |
| v0.2 | Feedback-loop experiment: retrain 8 times with and without blocking, plus a fix | In progress |

## v0.1

```
Kaggle data → train (XGBoost vs PyTorch) → MLflow registry → Docker + FastAPI → Cloud Run
```

- **Training:** XGBoost and a PyTorch neural net, trained on older transactions and tested on newer ones.
- **Versioning:** every model is registered in MLflow. A new model is promoted only if it beats the current one.
- **Serving:** FastAPI on Cloud Run. `/predict` scores transactions, `/health` shows the model version.
- **Monitoring:** requests are logged. `/monitoring` reports drift (PSI) against the training data.
- **CI:** tests run on every push.

| Model | Test PR-AUC |
|---|---|
| Random guess | 0.034 |
| PyTorch | 0.441 |
| **XGBoost (served)** | **0.487** |

## Data

[IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) (Kaggle): about
590,000 card transactions over six months, 3.5% fraud.

## Run

```bash
pip install -r requirements.txt
python src/train.py                                   # train, register, export the best model
python -m uvicorn serve:app --app-dir src --port 8080 # serve locally
pytest
```
