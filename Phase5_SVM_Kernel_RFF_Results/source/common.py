#!/usr/bin/env python3
"""
common.py - shared, dependency-light helpers for the Phase 5 remediation stages.

Everything that must behave identically across stages and tests lives here:

* strict frozen-scaler loading (never fits, accepts both key conventions)
* canonical Scope A/B metric functions (transition, High-entry, Brier, log-loss)
* the validation-only selection rules (tie rule, RFF seed policy)
* deterministic temporal-block training subsets (Stage B)
* timestamp-based block ids and a fast paired block bootstrap
* a tiny append-only run registry

Nothing in here reads test data or touches Phase 2-4 files.
"""
from __future__ import annotations

import json
import os
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

CLASS_LABELS: Tuple[int, int, int] = (0, 1, 2)
CLASS_NAMES = {0: "Low", 1: "Moderate", 2: "High"}
EPS_LOGLOSS = 1e-12
TIE_TOL = 0.005  # practical decision convention, NOT a significance test
DEFAULT_SEEDS = (42, 43, 44, 45, 46)


# --------------------------------------------------------------------------
# Paths / environment
# --------------------------------------------------------------------------
def package_root() -> Path:
    """Directory that contains source/ (works wherever the package is unzipped)."""
    return Path(__file__).resolve().parent.parent


def peak_rss_mb() -> float:
    """Peak resident set size of this process in MB (Linux: KB, macOS: bytes)."""
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v / (1024 ** 2) if sys.platform == "darwin" else v / 1024.0


def current_rss_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    except Exception:  # pragma: no cover
        return float("nan")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# Frozen scaler (Stage A artifact format: mean, scale, var, n_samples_seen)
# --------------------------------------------------------------------------
class ScalerError(RuntimeError):
    """Raised when a frozen scaler cannot be loaded.  There is NO fallback fit."""


def load_frozen_scaler(path: os.PathLike) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load (mean, scale) from a scaler NPZ written by Stage A.

    Accepted key conventions:  ``mean``/``scale``  (what Stage A writes)  and the
    legacy sklearn-attribute spelling ``mean_``/``scale_``.  If the file is missing
    or has neither convention this raises ScalerError.  It never fits a scaler
    (the previous Stage B silently fitted one on the data it was handed).
    """
    p = Path(path)
    if not p.exists():
        raise ScalerError(
            f"Frozen scaler not found: {p}. Run Stage A first; refusing to fit a replacement "
            "scaler on validation/test or on any other array."
        )
    with np.load(p, allow_pickle=False) as z:
        keys = set(z.files)
        for mk, sk in (("mean", "scale"), ("mean_", "scale_")):
            if mk in keys and sk in keys:
                mean, scale = np.asarray(z[mk], dtype=np.float64), np.asarray(z[sk], dtype=np.float64)
                break
        else:
            raise ScalerError(f"Scaler {p} has keys {sorted(keys)}; expected mean/scale or mean_/scale_.")
    if mean.shape != scale.shape or mean.ndim != 1:
        raise ScalerError(f"Scaler {p}: inconsistent shapes {mean.shape} vs {scale.shape}")
    if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(scale)) and np.all(scale > 0)):
        raise ScalerError(f"Scaler {p}: non-finite mean/scale or non-positive scale")
    return mean, scale


def apply_frozen_scaler(F: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """(F - mean) / scale in float64 (explicit dtype transition float32 -> float64)."""
    if F.shape[1] != mean.shape[0]:
        raise ScalerError(f"Feature dim {F.shape[1]} != scaler dim {mean.shape[0]}")
    return (np.asarray(F, dtype=np.float64) - mean) / scale


# --------------------------------------------------------------------------
# Canonical metrics (identical semantics to s06 / s03; tested for equality)
# --------------------------------------------------------------------------
def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Shared zero-denominator convention: undefined precision/recall/F1 -> 0.0."""
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def confusion3(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    yhat = np.asarray(yhat, dtype=np.int64)
    return np.bincount(y * 3 + yhat, minlength=9).reshape(3, 3)


def macro_f1_from_conf(C: np.ndarray) -> float:
    """Macro-F1 over the 3 fixed labels (a class with zero support & zero predictions counts as 0)."""
    C = np.asarray(C, dtype=np.float64)
    tp = np.diag(C)
    prec_den, rec_den = C.sum(axis=0), C.sum(axis=1)
    f = np.zeros(3)
    for k in range(3):
        p = tp[k] / prec_den[k] if prec_den[k] > 0 else 0.0
        r = tp[k] / rec_den[k] if rec_den[k] > 0 else 0.0
        f[k] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return float(f.mean())


def balanced_accuracy_from_conf(C: np.ndarray) -> float:
    C = np.asarray(C, dtype=np.float64)
    rec = [C[k, k] / C[k].sum() for k in range(3) if C[k].sum() > 0]
    return float(np.mean(rec)) if rec else 0.0


def classification_metrics(y: np.ndarray, yhat: np.ndarray) -> Dict[str, float]:
    C = confusion3(y, yhat)
    n = int(C.sum())
    out = {"n": n, "accuracy": float(np.trace(C) / n) if n else float("nan"),
           "macro_f1": macro_f1_from_conf(C), "balanced_accuracy": balanced_accuracy_from_conf(C)}
    w = C.sum(axis=1) / max(n, 1)
    per = []
    for k in range(3):
        p = C[k, k] / C[:, k].sum() if C[:, k].sum() > 0 else 0.0
        r = C[k, k] / C[k].sum() if C[k].sum() > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        out[f"precision_{k}"], out[f"recall_{k}"], out[f"f1_{k}"] = float(p), float(r), float(f)
        per.append(f)
    out["weighted_f1"] = float(np.dot(w, per))
    return out


def transition_metrics(y: np.ndarray, yhat: np.ndarray, c: np.ndarray) -> Dict[str, float]:
    """
    Scope A transition + High-entry metrics.  Rows with c < 0 are dropped (Scope A only).
    T = 1[y != c], That = 1[yhat != c];  H = 1[c != 2 & y == 2], Hhat = 1[c != 2 & yhat == 2].
    """
    m = np.asarray(c) >= 0
    y, yhat, c = np.asarray(y)[m], np.asarray(yhat)[m], np.asarray(c)[m]
    T, Th = (y != c), (yhat != c)
    tp, fp, fn, tn = int((T & Th).sum()), int((~T & Th).sum()), int((T & ~Th).sum()), int((~T & ~Th).sum())
    p, r, f = _prf(tp, fp, fn)
    out = {"n_scope_a": int(m.sum()), "true_transitions": int(T.sum()), "pred_transitions": int(Th.sum()),
           "trans_tp": tp, "trans_fp": fp, "trans_fn": fn, "trans_tn": tn,
           "trans_precision": p, "trans_recall": r, "trans_f1": f,
           "trans_dest_accuracy": float((yhat[T] == y[T]).mean()) if T.any() else 0.0,
           "trans_stable_accuracy": float((yhat[~T] == y[~T]).mean()) if (~T).any() else 0.0}
    H, Hh = (c != 2) & (y == 2), (c != 2) & (yhat == 2)
    htp, hfp, hfn = int((H & Hh).sum()), int((~H & Hh).sum()), int((H & ~Hh).sum())
    hp, hr, hf = _prf(htp, hfp, hfn)
    out.update({"true_high_entries": int(H.sum()), "pred_high_entries": int(Hh.sum()),
                "high_tp": htp, "high_fp": hfp, "high_fn": hfn,
                "high_precision": hp, "high_recall": hr, "high_f1": hf})
    for a in range(3):
        for b in range(3):
            if a != b:
                out[f"dir_{a}to{b}_true"] = int(((c == a) & (y == b)).sum())
                out[f"dir_{a}to{b}_correct"] = int(((c == a) & (y == b) & (yhat == b)).sum())
    return out


def prob_metrics(y: np.ndarray, P: np.ndarray, eps: float = EPS_LOGLOSS) -> Dict[str, float]:
    """
    Unscaled multiclass Brier  BS = mean_i sum_k (p_ik - 1[y_i=k])^2   and
    log-loss = -mean_i log p_i,y_i, both on probabilities clipped to [eps, 1] and row-renormalised.
    """
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"Expected N x 3 probabilities, got {P.shape}")
    Pc = np.clip(P, eps, 1.0)
    Pc = Pc / Pc.sum(axis=1, keepdims=True)
    y = np.asarray(y, dtype=np.int64)
    onehot = np.eye(3)[y]
    return {"brier": float(np.mean(np.sum((Pc - onehot) ** 2, axis=1))),
            "log_loss": float(-np.mean(np.log(Pc[np.arange(len(y)), y])))}


def reliability_table(y: np.ndarray, P: np.ndarray, n_bins: int = 10, cls: int = 2) -> pd.DataFrame:
    """Equal-width reliability bins for one class (default: High)."""
    p = np.asarray(P)[:, cls]
    obs = (np.asarray(y) == cls).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        s = idx == b
        rows.append({"bin": b, "lo": edges[b], "hi": edges[b + 1], "n": int(s.sum()),
                     "mean_pred": float(p[s].mean()) if s.any() else np.nan,
                     "obs_freq": float(obs[s].mean()) if s.any() else np.nan})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Validation-only selection rules
# --------------------------------------------------------------------------
def select_with_tie_rule(rows: List[dict], primary: str = "val_scopeB_macro_f1", tol: float = TIE_TOL,
                         secondary: Sequence[str] = ("val_trans_f1", "val_trans_recall", "val_scopeB_balanced_accuracy"),
                         cost_key: str = "cost") -> dict:
    """
    Deterministic selection.  Among candidates whose primary score is within ``tol`` of the best,
    pick the one with the highest secondary criteria in order (Scope A transition F1, transition recall,
    balanced accuracy), then the lowest cost, then the lowest input position.
    'Within tol' is a practical decision convention, not evidence of equivalence.
    """
    if not rows:
        raise ValueError("no candidates")
    best = max(r[primary] for r in rows)
    pool = [(i, r) for i, r in enumerate(rows) if best - r[primary] <= tol]
    pool.sort(key=lambda ir: (tuple(-float(ir[1].get(k, 0.0)) for k in secondary),
                              float(ir[1].get(cost_key, 0.0)), ir[0]))
    return pool[0][1]


def select_rff_policy(df: pd.DataFrame, configs: Sequence[str] = ("pca_B", "hosvd_B"),
                      score: str = "val_scopeB_macro_f1") -> dict:
    """
    Pre-declared, validation-only RFF selection rule ("median-seed common pair"):

      1. D*  = the D with the highest mean validation macro-F1 over {configs} x {all seeds};
               ties -> smaller D.
      2. For each seed s: m(s) = mean over configs of validation macro-F1 at (D*, s).
      3. Rank seeds by m(s); seed* = the median-ranked seed (lower median if the count is even);
         ties in m(s) -> smaller seed.
      4. Both configs then use (D*, seed*) -> identical random frequencies/phases for the pair
         (valid because d and gamma coincide), and gamma, C are taken from the stored experiment rows.

    The rule never chooses the best or the worst seed; all seeds stay in the results tables.
    """
    d = df[df["config"].isin(configs)]
    mean_by_D = d.groupby("D")[score].mean()
    D_star = int(sorted(mean_by_D.index, key=lambda D: (-mean_by_D[D], D))[0])
    sub = d[d["D"] == D_star]
    m = sub.groupby("seed")[score].mean()
    order = sorted(m.index, key=lambda s: (-m[s], s))            # descending score
    seed_star = int(order[(len(order) - 1) // 2])                # lower median of the ranking
    chosen = {}
    for cfg in configs:
        r = sub[(sub["config"] == cfg) & (sub["seed"] == seed_star)].iloc[0]
        chosen[cfg] = {"config": cfg, "D": D_star, "seed": seed_star, "gamma": float(r["gamma"]),
                       "C": float(r["C"]), "val_scopeB_macro_f1": float(r[score]),
                       "val_scopeA_trans_f1": float(r.get("val_trans_f1", np.nan))}
    return {"rule": "median-seed common pair (see select_rff_policy docstring)", "D_star": D_star,
            "seed_star": seed_star, "mean_val_macro_f1_by_D": {int(k): float(v) for k, v in mean_by_D.items()},
            "pair_mean_val_macro_f1_by_seed_at_Dstar": {int(k): float(v) for k, v in m.items()},
            "seed_rank_best_to_worst": [int(s) for s in order], "chosen": chosen}


# --------------------------------------------------------------------------
# Stage B deterministic temporal-block subsets
# --------------------------------------------------------------------------
def temporal_block_subset(n_total: int, n_fit: int, block_len: int = 100) -> np.ndarray:
    """
    Deterministic, training-only, systematic temporal-block subset.

    Rows are cut into consecutive blocks of ``block_len`` rows (the last partial block is never used),
    and ``n_fit / block_len`` blocks are taken at evenly spaced positions across the WHOLE training
    period (so every season/year is represented).  Returns sorted row indices (chronological order is
    preserved).  Identical indices are used for every representation.
    """
    if n_fit % block_len:
        raise ValueError("n_fit must be a multiple of block_len")
    n_full = n_total // block_len
    k = n_fit // block_len
    if k > n_full:
        raise ValueError("subset larger than training set")
    ids = np.unique(np.round(np.linspace(0, n_full - 1, k)).astype(int))
    if len(ids) != k:  # pragma: no cover  (only when k ~ n_full)
        ids = np.arange(k)
    return np.concatenate([np.arange(i * block_len, (i + 1) * block_len) for i in ids])


# --------------------------------------------------------------------------
# Timestamp-based blocks + paired block bootstrap (sufficient statistics)
# --------------------------------------------------------------------------
def calendar_block_ids(ts: np.ndarray, hours: int) -> np.ndarray:
    """
    Block id = floor(elapsed real time since first timestamp / hours).  Based on actual timestamps,
    so missing hours do not stretch a block; empty blocks simply do not occur.  Ids are re-labelled 0..B-1.
    """
    ts = np.asarray(ts).astype("datetime64[ns]")
    raw = ((ts - ts.min()) // np.timedelta64(hours, "h")).astype(np.int64)
    _, ids = np.unique(raw, return_inverse=True)
    return ids


def _block_stats(y, yhat, c, ids, n_blocks):
    """Per-block sufficient statistics: confusion(9), trans tp/fp/fn, high tp/fp/fn."""
    y, yhat, c = np.asarray(y), np.asarray(yhat), np.asarray(c)
    out = np.zeros((n_blocks, 15))
    conf = ids * 9 + (y * 3 + yhat)
    out[:, :9] = np.bincount(conf, minlength=n_blocks * 9).reshape(n_blocks, 9)
    a = c >= 0
    T, Th = (y != c) & a, (yhat != c) & a
    H, Hh = a & (c != 2) & (y == 2), a & (c != 2) & (yhat == 2)
    for j, v in enumerate([T & Th, a & ~T & Th, T & ~Th, H & Hh, a & ~H & Hh, H & ~Hh]):
        out[:, 9 + j] = np.bincount(ids, weights=v.astype(float), minlength=n_blocks)
    return out


def _metrics_from_stats(S: np.ndarray) -> Dict[str, np.ndarray]:
    """S: (B, 15) summed statistics -> dict of arrays (macro_f1, trans_f1, high_f1, high_recall...)."""
    B = S.shape[0]
    macro = np.array([macro_f1_from_conf(S[b, :9].reshape(3, 3)) for b in range(B)])

    def f1(tp, fp, fn):
        p = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        r = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
        f = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
        return p, r, f
    tp_, tr_, tf_ = f1(S[:, 9], S[:, 10], S[:, 11])
    hp_, hr_, hf_ = f1(S[:, 12], S[:, 13], S[:, 14])
    return {"macro_f1": macro, "trans_precision": tp_, "trans_recall": tr_, "trans_f1": tf_,
            "high_precision": hp_, "high_recall": hr_, "high_f1": hf_}


def paired_block_bootstrap(y, c, preds: Dict[str, np.ndarray], ts, pairs: Sequence[Tuple[str, str]],
                           block_hours: int = 24, n_boot: int = 1000, seed: int = 42,
                           metrics: Sequence[str] = ("macro_f1", "trans_f1", "high_f1")) -> dict:
    """
    Paired moving-block-style bootstrap on real-time blocks: blocks are resampled with replacement
    (same number as observed), the SAME resampled blocks are used for every model, and
    metric differences are formed on each resample.  Percentile 95% intervals.

    This quantifies test-set sampling variability under block resampling only.  It does NOT include
    training-set variability or RFF random-seed variability, and block dependence beyond
    ``block_hours`` is ignored.
    """
    ids = calendar_block_ids(ts, block_hours)
    nb = int(ids.max()) + 1
    stats = {k: _block_stats(y, p, c, ids, nb) for k, p in preds.items()}
    rng = np.random.default_rng(seed)
    mult = rng.multinomial(nb, np.full(nb, 1.0 / nb), size=n_boot).astype(float)  # (B, nb)
    boot = {k: _metrics_from_stats(mult @ S) for k, S in stats.items()}
    obs = {k: _metrics_from_stats(S.sum(axis=0, keepdims=True)) for k, S in stats.items()}
    res = {"block_hours": block_hours, "n_blocks": nb, "n_boot": n_boot, "seed": seed,
           "n_rows": int(len(y)), "pairs": {}}
    for a, b in pairs:
        r = {}
        for m in metrics:
            d = boot[a][m] - boot[b][m]
            r[m] = {"observed_a": float(obs[a][m][0]), "observed_b": float(obs[b][m][0]),
                    "observed_diff_a_minus_b": float(obs[a][m][0] - obs[b][m][0]),
                    "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                    "se": float(d.std(ddof=1)),
                    "ci_includes_zero": bool(np.percentile(d, 2.5) <= 0 <= np.percentile(d, 97.5))}
        res["pairs"][f"{a} - {b}"] = r
    return res


# --------------------------------------------------------------------------
# Run registry (append-only JSON lines)
# --------------------------------------------------------------------------
class RunRegistry:
    def __init__(self, path: os.PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, **kw) -> None:
        kw.setdefault("utc", utc_now())
        with open(self.path, "a") as fh:
            fh.write(json.dumps(kw, default=str) + "\n")

    def read(self) -> List[dict]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]
