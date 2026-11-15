import numpy as np

from monitor import drift_report, make_reference, psi

rng = np.random.default_rng(0)


def test_same_distribution_has_low_psi():
    ref = make_reference(rng.normal(size=20_000))
    assert psi(ref, rng.normal(size=5_000)) < 0.02


def test_shifted_distribution_has_high_psi():
    ref = make_reference(rng.normal(size=20_000))
    assert psi(ref, rng.normal(loc=1.0, size=5_000)) > 0.2


def test_missingness_shift_is_detected():
    train = rng.normal(size=10_000)
    live = rng.normal(size=1_000)
    live[:500] = np.nan  # half the live values suddenly missing
    assert psi(make_reference(train), live) > 0.2


def test_drift_report_flags_only_shifted_columns():
    import pandas as pd
    ref = {"a": make_reference(rng.normal(size=10_000)), "b": make_reference(rng.normal(size=10_000))}
    live = pd.DataFrame({"a": rng.normal(size=2_000), "b": rng.normal(loc=2, size=2_000)})
    assert drift_report(ref, live)["drifted"] == ["b"]
