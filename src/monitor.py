"""Drift monitoring: compare live feature/score distributions with training using PSI (population stability index)."""
import numpy as np
import pandas as pd

PSI_ALERT = 0.2  # common rule of thumb: < 0.1 stable, 0.1-0.2 watch, > 0.2 shifted


def _fractions(edges: np.ndarray, values) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    idx = np.searchsorted(edges, v, side="right")
    idx[np.isnan(v)] = len(edges) + 1  # missing values get their own bin: missingness can drift too
    return np.bincount(idx, minlength=len(edges) + 2) / max(len(v), 1)


def make_reference(values) -> dict:
    """Decile bin edges of the training values, and the share of training values in each bin."""
    v = np.asarray(values, dtype=float)
    present = v[~np.isnan(v)]
    edges = np.unique(np.quantile(present, np.linspace(0.1, 0.9, 9))) if present.size else np.array([])
    return {"edges": edges.tolist(), "fractions": _fractions(edges, v).tolist()}


def psi(reference: dict, values, eps: float = 1e-4) -> float:
    expected = np.asarray(reference["fractions"]) + eps
    actual = _fractions(np.asarray(reference["edges"]), values) + eps
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def drift_report(reference: dict, current: pd.DataFrame) -> dict:
    scores = {col: round(psi(ref, current[col]), 4) for col, ref in reference.items() if col in current}
    return {
        "n_requests": len(current),
        "psi": dict(sorted(scores.items(), key=lambda kv: -kv[1])),
        "drifted": [c for c, s in scores.items() if s > PSI_ALERT],
    }
