"""
fixed_fusion.py – Lambda convex mixture p_F = lambda*p_S + (1-lambda)*p_M.
Endpoints: lambda=0 => Markov; lambda=1 => SVM.
"""
import numpy as np
from evaluation import compute_metrics


def grid_candidates(lam_grid):
    return [float(l) for l in lam_grid]


def fuse_one_row(p_m, p_s, lam):
    return np.asarray(lam) * np.asarray(p_s) + (1.0 - lam) * np.asarray(p_m)


def run_fixed_fusion(p_markov, p_svm, y_true, current_state, lam_grid, scope_label):
    p_markov = np.asarray(p_markov, dtype=np.float64)
    p_svm    = np.asarray(p_svm, dtype=np.float64)
    results = []
    # endpoint sanity checks (lambda=0 => markov; lambda=1 => svm) on Scope A only
    mask_a = np.asarray(current_state) >= 0
    check0 = np.allclose(p_markov[mask_a], fuse_one_row(p_markov, p_svm, 0.0)[mask_a])
    check1 = np.allclose(p_svm[mask_a],    fuse_one_row(p_markov, p_svm, 1.0)[mask_a])
    assert check0 and check1, "Endpoint identity violated"
    for lam in lam_grid:
        fused = fuse_one_row(p_markov, p_svm, lam)
        preds = np.argmax(fused, axis=1)
        met = compute_metrics(y_true, preds, current_state, probs=fused)
        met.update({"model": f"fixed_lambda_{lam}", "fusion_type": "fixed",
                     "lambda": lam, "scope": scope_label})
        results.append(met)
    return results
