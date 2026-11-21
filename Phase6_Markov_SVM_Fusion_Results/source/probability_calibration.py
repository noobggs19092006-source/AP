"""
probability_calibration.py – Train-only calibration diagnostics.
Audit existing calibrators; no new labels used for fitting here.
"""
import numpy as np
from evaluation import ensure_prob_simplex, compute_metrics


def audit_calibrator(probs, y_true, current_state, scope_label):
    """Verify saved probabilities are properly calibrated simplex vectors."""
    probs = np.asarray(probs, dtype=np.float64)
    assert probs.shape[1] == 3, f"Expected shape [N,3], got {probs.shape}"
    assert not np.any(np.isnan(probs)), f"NaN probabilities in {scope_label}"
    sums = probs.sum(axis=1)
    bad = int(np.sum((sums < 0.999) | (sums > 1.001)))
    if bad:
        raise ValueError(f"{scope_label}: {bad}/{len(probs)} rows do not sum to ≈1")
    nonneg = int(np.sum(probs < 0))
    if nonneg:
        raise ValueError(f"{scope_label}: {nonneg} negative probabilities")
    met = compute_metrics(y_true, np.argmax(probs, axis=1), current_state, probs=probs, scope_label=scope_label)
    return met
