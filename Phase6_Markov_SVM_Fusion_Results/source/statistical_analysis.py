"""
statistical_analysis.py – Paired time-block bootstrap CI.
"""
import numpy as np


def block_bootstrap(paired_diffs, block_length=24, n_resamples=1000, seed=42):
    """Paired difference array; resample contiguous blocks of given length."""
    N = len(paired_diffs)
    rng = np.random.default_rng(seed)
    diffs = np.asarray(paired_diffs, dtype=float)
    if diffs.size == 0:
        return np.array([np.nan]), np.array([np.nan]), np.array([np.nan])
    blocks = []
    i = 0
    while i < N:
        end = min(i + block_length, N)
        blocks.append(diffs[i:end])
        i = end
    if not blocks:
        blocks = [diffs[:]]
    bs_means = np.empty(n_resamples, dtype=float)
    for r in range(n_resamples):
        idx = rng.integers(0, len(blocks), size=N // block_length + 1)
        sample = np.concatenate([blocks[k] for k in idx[:N // block_length + 1]])
        bs_means[r] = sample[:N].mean()
    lo = float(np.percentile(bs_means, 2.5))
    hi = float(np.percentile(bs_means, 97.5))
    point = float(diffs.mean())
    return point, lo, hi
