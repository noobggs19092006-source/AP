"""
adaptive_fusion.py – Transition-risk-dependent gating.
lambda_t = lambda_high if r_S,t >= tau else lambda_low.
"""
import numpy as np
from evaluation import compute_metrics


def build_adaptive_masks(p_svm, current_state, tau, lambda_low, lambda_high):
    r_S = 1.0 - np.take_along_axis(p_svm, current_state[:, None].astype(int), axis=1).squeeze(-1)
    mask_high = r_S >= tau
    return mask_high


def fuse_adaptive(p_markov, p_svm, mask_high, lambda_low, lambda_high):
    lambdas = np.where(mask_high, lambda_high, lambda_low)
    p_out = np.empty_like(p_svm)
    for j in range(3):
        p_out[:, j] = lambdas * p_svm[:, j] + (1.0 - lambdas) * p_markov[:, j]
    # renormalize to handle floating drift
    s = p_out.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return p_out / s, lambdas


def run_adaptive_grid(p_markov, p_svm, y_true, current_state, tau_grid, lambda_low_grid, lambda_high_grid):
    rows = []
    for tau in tau_grid:
        for ll in lambda_low_grid:
            for lh in lambda_high_grid:
                if ll > lh:
                    continue
                mask_h = build_adaptive_masks(p_svm, current_state, tau, ll, lh)
                fused, lambdas = fuse_adaptive(p_markov, p_svm, mask_h, ll, lh)
                preds = np.argmax(fused, axis=1)
                met = compute_metrics(y_true, preds, current_state, probs=fused)
                met.update({"model": f"adaptive_tau{tau}_low{ll}_high{lh}",
                             "fusion_type": "adaptive", "tau": tau,
                             "lambda_low": ll, "lambda_high": lh,
                             "n_branch_high": int(mask_h.sum()),
                             "n_branch_low": int((~mask_h).sum()),
                             "scope": "val"})
                rows.append(met)
    return rows
