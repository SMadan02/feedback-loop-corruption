"""v0.1 baseline: train XGBoost and a PyTorch net on a time split, version both in the MLflow registry,
promote the better one to 'champion' only if it beats the current champion, and export it to model/ for serving.

Run from anywhere:  .venv/Scripts/python src/train.py
"""
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from monitor import make_reference
from scorer import FraudScorer, nn_inputs, nn_stats

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "fraud-detector"
SEED = 0


def load(features: list[str]) -> pd.DataFrame:
    cols = ["TransactionDT", "isFraud", *features]
    df = pd.read_csv(ROOT / "data/raw/train_transaction.csv.zip", usecols=cols)
    return df.sort_values("TransactionDT", ignore_index=True)


def time_split(df, valid_share, test_share):
    """Oldest rows train, newer rows validate, newest rows test: mimics deploying on the future."""
    a, b = int(len(df) * (1 - valid_share - test_share)), int(len(df) * (1 - test_share))
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]


def train_xgboost(Xtr, ytr, Xva, yva) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=2000, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", eval_metric="aucpr", early_stopping_rounds=100, random_state=SEED,
    )
    return model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)


def train_pytorch(Xtr, ytr, Xva, yva, stats, max_epochs=15, patience=3):
    torch.manual_seed(SEED)
    xtr, xva = torch.from_numpy(nn_inputs(Xtr, stats)), torch.from_numpy(nn_inputs(Xva, stats))
    t = torch.tensor(ytr.to_numpy(), dtype=torch.float32).unsqueeze(1)
    net = torch.nn.Sequential(
        torch.nn.Linear(xtr.shape[1], 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 32), torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()  # plain log loss, same as XGBoost: no imbalance re-weighting
    best, best_state, bad = -1.0, None, 0
    for epoch in range(max_epochs):
        net.train()
        for idx in torch.randperm(len(xtr)).split(2048):
            opt.zero_grad()
            loss_fn(net(xtr[idx]), t[idx]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val = average_precision_score(yva, net(xva).squeeze(1).numpy())
        print(f"  pytorch epoch {epoch}: valid PR-AUC {val:.4f}")
        if val > best:
            best, best_state, bad = val, {k: v.clone() for k, v in net.state_dict().items()}, 0
        elif (bad := bad + 1) >= patience:  # early stopping on validation PR-AUC
            break
    net.load_state_dict(best_state)
    return net.eval()


def current_champion_pr_auc(client) -> float:
    try:
        return float(client.get_model_version_by_alias(MODEL_NAME, "champion").tags["test_pr_auc"])
    except MlflowException:  # no champion yet: anything beats nothing
        return -1.0


def main():
    cfg_all = yaml.safe_load((ROOT / "config.yaml").read_text())
    cfg, features = cfg_all["baseline"], cfg_all["baseline"]["features"]
    block_share = cfg_all["simulation"]["blocking_share"]

    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    if mlflow.get_experiment_by_name("p1-baseline") is None:
        mlflow.create_experiment("p1-baseline", artifact_location=(ROOT / "mlruns").as_uri())
    mlflow.set_experiment("p1-baseline")
    client = MlflowClient()

    df = load(features)
    tr, va, te = time_split(df, cfg["valid_share"], cfg["test_share"])
    Xtr, Xva, Xte = tr[features], va[features], te[features]
    ytr, yva, yte = tr["isFraud"], va["isFraud"], te["isFraud"]
    print(f"rows: train {len(tr):,} | valid {len(va):,} | test {len(te):,} | test fraud rate {yte.mean():.4f}")

    stats = nn_stats(Xtr)
    candidates = {
        "xgboost": lambda: FraudScorer("xgboost", train_xgboost(Xtr, ytr, Xva, yva), features),
        "pytorch": lambda: FraudScorer("pytorch", train_pytorch(Xtr, ytr, Xva, yva, stats), features, stats),
    }
    results = {}
    for kind, build in candidates.items():
        print(f"training {kind} ...")
        with mlflow.start_run(run_name=kind):
            scorer = build()
            train_scores = scorer.score(Xtr)
            scorer.block_threshold = float(np.quantile(train_scores, 1 - block_share))
            scorer.reference = {c: make_reference(Xtr[c]) for c in features}
            scorer.reference["score"] = make_reference(train_scores)

            pr_auc = average_precision_score(yte, scorer.score(Xte))
            mlflow.log_params({"kind": kind, "n_features": len(features), "block_share": block_share,
                               "train_rows": len(tr), "seed": SEED})
            mlflow.log_metrics({"test_pr_auc": pr_auc, "test_fraud_rate": float(yte.mean()),
                                "valid_pr_auc": average_precision_score(yva, scorer.score(Xva))})
            info = mlflow.pyfunc.log_model(name="model", python_model=scorer,
                                           code_paths=[str(ROOT / "src/scorer.py")],
                                           registered_model_name=MODEL_NAME)
            version = info.registered_model_version
            client.set_model_version_tag(MODEL_NAME, version, "test_pr_auc", f"{pr_auc:.6f}")
            client.set_model_version_tag(MODEL_NAME, version, "kind", kind)
            results[kind] = (pr_auc, version)
            print(f"  {kind}: test PR-AUC {pr_auc:.4f} (random model = {yte.mean():.4f}) -> registered v{version}")

    # Promotion gate: the best new model replaces the champion only if it scores higher on the same test period.
    best_kind, (best_pr_auc, best_version) = max(results.items(), key=lambda kv: kv[1][0])
    old = current_champion_pr_auc(client)
    if best_pr_auc > old:
        client.set_registered_model_alias(MODEL_NAME, "champion", best_version)
        print(f"promoted v{best_version} ({best_kind}, {best_pr_auc:.4f}) over previous champion ({old:.4f})")
    else:
        print(f"kept existing champion ({old:.4f}); best new model scored {best_pr_auc:.4f}")

    export(client)


def export(client):
    """Copy the champion out of the registry into model/, which the Docker image serves."""
    champ = client.get_model_version_by_alias(MODEL_NAME, "champion")
    out = ROOT / "model"
    shutil.rmtree(out, ignore_errors=True)
    mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{MODEL_NAME}@champion", dst_path=str(out))
    (out / "version.json").write_text(json.dumps({
        "name": MODEL_NAME, "version": champ.version, "kind": champ.tags["kind"],
        "test_pr_auc": float(champ.tags["test_pr_auc"]),
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2))
    print(f"exported champion v{champ.version} to {out}")


if __name__ == "__main__":
    main()
