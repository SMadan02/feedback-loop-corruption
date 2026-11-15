"""The served model: one interface for both model kinds. Raw feature columns in, fraud probability out."""
import mlflow.pyfunc
import numpy as np
import pandas as pd


def _signed_log(X: pd.DataFrame) -> np.ndarray:
    v = X.to_numpy(dtype=float)
    return np.sign(v) * np.log1p(np.abs(v))  # tames heavy tails (amounts, C counts) for the neural net


def nn_stats(X: pd.DataFrame) -> dict:
    z = _signed_log(X)
    mean, std = np.nanmean(z, axis=0), np.nanstd(z, axis=0)
    return {"mean": np.nan_to_num(mean), "std": np.where(np.nan_to_num(std) > 0, std, 1.0)}


def nn_inputs(X: pd.DataFrame, stats: dict) -> np.ndarray:
    """Standardised values (missing -> 0, i.e. the mean) plus a 0/1 'was missing' column per feature."""
    z = (_signed_log(X) - stats["mean"]) / stats["std"]
    missing = np.isnan(z)
    return np.hstack([np.nan_to_num(np.clip(z, -5, 5)), missing]).astype(np.float32)


class FraudScorer(mlflow.pyfunc.PythonModel):
    """Wraps an XGBoost classifier or a PyTorch net so serving never needs to know which one won."""

    def __init__(self, kind: str, model, features: list[str], stats: dict | None = None):
        self.kind, self.model, self.features, self.stats = kind, model, features, stats
        self.block_threshold = None  # score at/above which a transaction is blocked (set in training)
        self.reference = None        # training-data distributions for drift monitoring (set in training)

    def score(self, X: pd.DataFrame) -> np.ndarray:
        X = X.reindex(columns=self.features).astype(float)  # missing columns -> NaN, extra columns dropped
        if self.kind == "xgboost":
            return self.model.predict_proba(X)[:, 1]
        import torch  # only needed when the PyTorch model is the one being served
        with torch.no_grad():
            logits = self.model(torch.from_numpy(nn_inputs(X, self.stats)))
        return torch.sigmoid(logits).squeeze(1).numpy().astype(float)

    def predict(self, context, model_input, params=None):
        return self.score(model_input)
