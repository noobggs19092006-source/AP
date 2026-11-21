"""markov_core.py -- pure, dataset-independent building blocks for Phase 3.

Everything here works on small integer arrays so it can be unit-tested
without the Phase 2 data.  State encoding: 0=low, 1=moderate, 2=high,
-1 = unobserved (only allowed for the *previous-hour* state).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

K = 3
CLASSES = (0, 1, 2)
CLASS_NAMES = ("low", "moderate", "high")
EPS_LOGLOSS = 1e-12          # fixed, disclosed; never tuned on any split
ROW_TOL = 1e-12              # row-stochastic tolerance (atol, rtol=0)

SEASON_OF_MONTH = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                   6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}


# --------------------------------------------------------------------------
# counts, MLE, tie-break
# --------------------------------------------------------------------------
def counts_from_pairs(prev: np.ndarray, nxt: np.ndarray, k: int = K) -> np.ndarray:
    """C[i,j] = #rows with prev==i and nxt==j; rows with prev<0 are skipped."""
    prev = np.asarray(prev, dtype=np.int64)
    nxt = np.asarray(nxt, dtype=np.int64)
    if prev.shape != nxt.shape:
        raise ValueError("prev and nxt must have the same shape")
    if np.any(nxt < 0) or np.any(nxt >= k):
        raise ValueError("next-state labels must lie in {0..k-1}")
    if np.any(prev >= k) or np.any(prev < -1):
        raise ValueError("previous-state labels must lie in {-1,0..k-1}")
    m = prev >= 0
    return np.bincount(prev[m] * k + nxt[m], minlength=k * k).reshape(k, k).astype(np.int64)


def counts_from_sequence(seq: Sequence[int]) -> np.ndarray:
    """Counts from a *contiguous* hourly sequence (tests / hand example only)."""
    s = np.asarray(seq, dtype=np.int64)
    return counts_from_pairs(s[:-1], s[1:])


def majority_label(class_counts: np.ndarray) -> int:
    """argmax of training label counts; smallest index on exact ties."""
    cc = np.asarray(class_counts)
    return int(np.flatnonzero(cc == cc.max())[0])


def fit_markov(counts: np.ndarray, class_counts: np.ndarray):
    """Unsmoothed MLE.  Zero-support rows fall back to the TRAIN label prior.

    Returns (P, row_totals, prior, fallback_rows).
    """
    counts = np.asarray(counts, dtype=np.int64)
    prior = np.asarray(class_counts, dtype=np.float64)
    prior = prior / prior.sum()
    row_n = counts.sum(axis=1)
    P = np.empty((K, K), dtype=np.float64)
    fallback_rows: List[int] = []
    for i in range(K):
        if row_n[i] > 0:
            P[i] = counts[i] / row_n[i]
        else:
            P[i] = prior
            fallback_rows.append(i)
    return P, row_n, prior, fallback_rows


def modal_rule(counts: np.ndarray, majority: int):
    """LOCKED deterministic modal rule per current state i.

    Tie set A = {j : P_ij == max_k P_ik}.  All entries of row i share the
    denominator n_i, so P_ij == P_ik  <=>  C_ij == C_ik: ties are decided by
    exact integer comparison, never by floating point or np.argmax.
    (a) i in A -> predict i;  (b) else predict min(A).
    A zero-support row has no counts: use the (train) majority label.
    Returns (pred_by_state, tie_sets) where tie_sets[i] is None for fallback rows.
    """
    counts = np.asarray(counts, dtype=np.int64)
    pred = np.empty(K, dtype=np.int64)
    ties: List[Optional[List[int]]] = []
    for i in range(K):
        row = counts[i]
        if row.sum() == 0:
            pred[i] = majority
            ties.append(None)
            continue
        m = int(row.max())
        A = [j for j in range(K) if int(row[j]) == m]
        pred[i] = i if i in A else min(A)
        ties.append(A)
    return pred, ties


# --------------------------------------------------------------------------
# predictors (hard label + probability rows), identical-row semantics
# --------------------------------------------------------------------------
def _check_cs(cs: np.ndarray) -> np.ndarray:
    cs = np.asarray(cs, dtype=np.int64)
    if np.any(cs < -1) or np.any(cs >= K):
        raise ValueError("current_state must lie in {-1,0,1,2}")
    return cs


def predict_majority(n: int, majority: int, prior: np.ndarray):
    return np.full(n, majority, dtype=np.int64), np.tile(prior, (n, 1))


def predict_persistence(cs: np.ndarray, majority: int, prior: np.ndarray):
    """Hard: current state if observed else TRAIN majority.
    Probability: degenerate one-hot (UNCALIBRATED) if observed else TRAIN prior."""
    cs = _check_cs(cs)
    known = cs >= 0
    pred = np.where(known, cs, majority)
    prob = np.tile(prior, (len(cs), 1))
    prob[known] = np.eye(K)[cs[known]]
    return pred, prob


def predict_markov(cs: np.ndarray, P: np.ndarray, modal_pred: np.ndarray,
                   majority: int, prior: np.ndarray):
    cs = _check_cs(cs)
    known = cs >= 0
    safe = np.where(known, cs, 0)
    pred = np.where(known, modal_pred[safe], majority)
    prob = np.where(known[:, None], P[safe], prior[None, :])
    return pred.astype(np.int64), prob


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def confusion(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Rows = true class, columns = predicted class, labels [0,1,2] always kept."""
    y = np.asarray(y, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    return np.bincount(y * K + pred, minlength=K * K).reshape(K, K).astype(np.int64)


def _sdiv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a/b with zero_division=0."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    out = np.zeros_like(a)
    np.divide(a, b, out=out, where=b > 0)
    return out


def classification_metrics(cm: np.ndarray) -> Dict[str, object]:
    """Metrics from a 3x3 confusion matrix (zero_division=0, class order [0,1,2]).

    macro_* average all three classes.  balanced_accuracy = mean recall over
    classes with support>0 (sklearn convention); balanced_accuracy_n_classes
    records how many classes contributed, so absent classes are never hidden.
    """
    cm = np.asarray(cm, dtype=np.int64)
    n = int(cm.sum())
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1)
    n_pred = cm.sum(axis=0)
    prec = _sdiv(tp, n_pred)
    rec = _sdiv(tp, support)
    f1 = _sdiv(2 * prec * rec, prec + rec)
    sup = support > 0
    return {
        "n": n,
        "accuracy": float(tp.sum() / n) if n else float("nan"),
        "macro_precision": float(prec.mean()),
        "macro_recall": float(rec.mean()),
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(rec[sup].mean()) if sup.any() else float("nan"),
        "balanced_accuracy_n_classes": int(sup.sum()),
        "precision": prec, "recall": rec, "f1": f1,
        "support": support.astype(np.int64), "n_pred": n_pred.astype(np.int64),
    }


def brier_score(prob: np.ndarray, y: np.ndarray) -> float:
    """Unnormalised multiclass Brier: mean_i sum_c (p_ic - 1[y_i=c])^2  (range 0..2)."""
    if len(y) == 0:
        return float("nan")
    oh = np.eye(K)[np.asarray(y, dtype=np.int64)]
    return float(np.mean(np.sum((prob - oh) ** 2, axis=1)))


def log_loss_clipped(prob: np.ndarray, y: np.ndarray, eps: float = EPS_LOGLOSS) -> Tuple[float, int]:
    """Mean negative log-likelihood (nats) with fixed clip eps then row renormalisation.
    Returns (loss, number of rows whose true class had probability exactly 0)."""
    if len(y) == 0:
        return float("nan"), 0
    y = np.asarray(y, dtype=np.int64)
    n_zero = int(np.sum(prob[np.arange(len(y)), y] == 0))
    p = np.clip(prob, eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return float(-np.mean(np.log(p[np.arange(len(y)), y]))), n_zero


def change_metrics(y: np.ndarray, cs: np.ndarray, pred: np.ndarray) -> Dict[str, object]:
    """Regime-change diagnostics; rows must all have observed cs>=0."""
    y = np.asarray(y); cs = np.asarray(cs); pred = np.asarray(pred)
    if np.any(cs < 0):
        raise ValueError("change metrics require observed current_state on every row")
    true_ch = y != cs
    pred_ch = pred != cs
    n = len(y)
    tp = int(np.sum(true_ch & pred_ch))
    n_true, n_pred = int(true_ch.sum()), int(pred_ch.sum())
    nan = float("nan")
    prec = tp / n_pred if n_pred else nan
    rec = tp / n_true if n_true else nan
    f1 = (2 * prec * rec / (prec + rec)) if (n_pred and n_true and (prec + rec) > 0) else nan
    corr_given_change = int(np.sum(true_ch & (pred == y)))
    enter = (cs != 2) & (y == 2)
    pred_enter = (cs != 2) & (pred == 2)
    n_enter = int(enter.sum())
    n_pe = int(pred_enter.sum())
    tp_enter = int((enter & (pred == 2)).sum())
    return {
        "n_evaluated": n, "n_true_change": n_true,
        "change_prevalence_pct": 100 * n_true / n if n else nan,
        "n_pred_change": n_pred, "n_change_detected_tp": tp,
        "change_precision": prec, "change_recall": rec, "change_f1": f1,
        "n_correct_next_state_given_change": corr_given_change,
        "next_state_accuracy_given_change": corr_given_change / n_true if n_true else nan,
        "n_enter_high_events": n_enter, "n_enter_high_predicted_correctly": tp_enter,
        "enter_high_recall": tp_enter / n_enter if n_enter else nan,
        "n_pred_enter_high": n_pe,
        "enter_high_precision": tp_enter / n_pe if n_pe else nan,
    }


# --------------------------------------------------------------------------
# Markov-chain linear algebra
# --------------------------------------------------------------------------
def is_row_stochastic(M: np.ndarray, atol: float = ROW_TOL) -> bool:
    M = np.asarray(M, dtype=np.float64)
    return bool(np.all(np.isfinite(M)) and np.all(M >= -atol)
                and np.allclose(M.sum(axis=1), 1.0, atol=atol, rtol=0))


def reachability(P: np.ndarray) -> np.ndarray:
    """Boolean transitive-closure (reflexive) of the positive-edge digraph."""
    k = len(P)
    R = (np.asarray(P) > 0) | np.eye(k, dtype=bool)
    for _ in range(int(math.ceil(math.log2(max(k, 2)))) + 1):
        R = (R.astype(np.int64) @ R.astype(np.int64)) > 0
    return R


def state_periods(P: np.ndarray) -> List[int]:
    """Per-state period = gcd{n : (A^n)_ii > 0}, n up to 2k^2 (0 if no return)."""
    A = (np.asarray(P) > 0).astype(np.int64)
    k = len(A)
    g = [0] * k
    M = np.eye(k, dtype=np.int64)
    for n in range(1, 2 * k * k + 1):
        M = ((M @ A) > 0).astype(np.int64)
        for i in range(k):
            if M[i, i]:
                g[i] = math.gcd(g[i], n)
    return g


def stationary_analysis(P: np.ndarray, tol: float = 1e-9) -> Dict[str, object]:
    """Stationary distribution (constrained least squares), eigen-spectrum,
    irreducibility / aperiodicity, and conditional spectral gap."""
    P = np.asarray(P, dtype=np.float64)
    k = len(P)
    A = np.vstack([P.T - np.eye(k), np.ones((1, k))])
    b = np.zeros(k + 1); b[-1] = 1.0
    pi, _, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    cond = float(sv.max() / sv.min()) if sv.min() > 0 else float("inf")
    resid = float(np.max(np.abs(pi @ P - pi)))
    out: Dict[str, object] = {
        "pi": pi, "constraint_matrix_rank": int(rank), "constraint_matrix_condition": cond,
        "stationary_residual_inf_norm": resid, "pi_sum_error": float(abs(pi.sum() - 1)),
        "pi_min": float(pi.min()), "pi_nonnegative_within_tol": bool(pi.min() >= -tol),
        "unique_solution_of_constrained_system": bool(rank == k),
    }
    # eigen-decomposition (right eigenvalues of P == left eigenvalues)
    w, _ = np.linalg.eig(P)
    order = np.lexsort((-w.real, -np.abs(w)))
    w = w[order]
    mult1 = int(np.sum(np.abs(w - 1) < tol))
    out["eigenvalues"] = w
    out["algebraic_multiplicity_of_1"] = mult1
    # independent left-eigenvector cross-check
    wl, Vl = np.linalg.eig(P.T)
    j = int(np.argmin(np.abs(wl - 1)))
    v = np.real(Vl[:, j]); v = v / v.sum()
    out["pi_left_eigenvector_check"] = v
    out["pi_vs_eigenvector_max_abs_diff"] = float(np.max(np.abs(v - pi)))
    # graph structure
    R = reachability(P)
    irreducible = bool(R.all())
    periods = state_periods(P)
    aperiodic = bool(irreducible and all(p == 1 for p in periods))
    out.update({"irreducible": irreducible, "state_periods": periods,
                "aperiodic_given_irreducible": aperiodic,
                "reachability_matrix": R})
    # spectral gap only under verified conditions
    nontriv = np.delete(w, int(np.argmin(np.abs(w - 1)))) if len(w) else w
    slem = float(np.max(np.abs(nontriv))) if len(nontriv) else float("nan")
    out["second_largest_eigenvalue_modulus"] = slem
    if irreducible and aperiodic and mult1 == 1:
        out["spectral_gap"] = 1.0 - slem
        out["spectral_gap_status"] = ("interpretable: chain irreducible, aperiodic, eigenvalue 1 simple; "
                                      "gap = 1 - max|nontrivial eigenvalue| (conditional on the fitted model)")
    else:
        out["spectral_gap"] = None
        out["spectral_gap_status"] = "not interpreted / conditions not met"
    return out


def expected_run_lengths(P: np.ndarray) -> List[float]:
    """Model-implied mean run length 1/(1-P_ii); inf if P_ii == 1."""
    return [float("inf") if P[i, i] >= 1 else 1.0 / (1.0 - P[i, i]) for i in range(len(P))]


# --------------------------------------------------------------------------
# timestamp handling: hourly series from aligned (current_state, y) rows
# --------------------------------------------------------------------------
def hour_index(ts: np.ndarray) -> np.ndarray:
    ts = np.asarray(ts).astype("datetime64[ns]")
    h = ts.astype("datetime64[h]")
    if not np.all(h.astype("datetime64[ns]") == ts):
        raise ValueError("timestamps must fall exactly on whole hours")
    return h.astype(np.int64)


def build_state_series(ts: np.ndarray, y: np.ndarray, cs: np.ndarray) -> Dict[str, object]:
    """Reconstruct the observed hourly state series of ONE split, using only
    the two *observed* channels: y at the target hour t and current_state at
    hour t-1.  Never shifts filtered arrays; unobserved hours stay -1.
    Never reads the (imputed) tensor X."""
    h = hour_index(ts)
    if len(h) and np.any(np.diff(h) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    y = np.asarray(y, dtype=np.int64); cs = _check_cs(cs)
    lo, hi = int(h.min()) - 1, int(h.max())
    s = -np.ones(hi - lo + 1, dtype=np.int64)
    s[h - lo] = y
    m = cs >= 0
    idx = h[m] - 1 - lo
    existing = s[idx]
    conflicts = int(np.sum((existing >= 0) & (existing != cs[m])))
    new_only = int(np.sum(existing < 0))
    s[idx] = np.where(existing >= 0, existing, cs[m])
    return {"series": s, "start_hour": lo, "n_conflicts": conflicts,
            "n_hours_from_current_state_only": new_only,
            "start_timestamp": np.datetime64(lo, "h")}


def triples_from_series(s: np.ndarray):
    """Valid (S[t-2], S[t-1], S[t]) with all three hours observed (exactly consecutive hours)."""
    a, b, c = s[:-2], s[1:-1], s[2:]
    m = (a >= 0) & (b >= 0) & (c >= 0)
    return a[m], b[m], c[m], np.flatnonzero(m) + 2


def strict_join_triples(ts: np.ndarray, y: np.ndarray, cs: np.ndarray):
    """Option 1 of the brief: row t gives (S[t-1]=cs, S[t]=y); S[t-2] is the
    current_state of the row whose target_timestamp is exactly t-1 hour."""
    h = hour_index(ts)
    cs = _check_cs(cs); y = np.asarray(y, dtype=np.int64)
    pos = np.searchsorted(h, h - 1)
    posc = np.minimum(pos, len(h) - 1)
    has_prev = (pos < len(h)) & (h[posc] == h - 1)
    ok = has_prev & (cs >= 0) & (cs[posc] >= 0)
    return cs[posc][ok], cs[ok], y[ok]


def counts3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.bincount(a * 9 + b * 3 + c, minlength=27).reshape(3, 3, 3).astype(np.int64)


def run_lengths(s: np.ndarray):
    """Run lengths of constant observed state inside contiguous observed segments.
    Returns list of (state, length, censored) where censored=True if the run touches
    a segment edge (start/end of data or adjacent to an unobserved hour)."""
    out = []
    n = len(s); i = 0
    while i < n:
        if s[i] < 0:
            i += 1; continue
        j = i
        while j + 1 < n and s[j + 1] == s[i]:
            j += 1
        censored = (i == 0 or s[i - 1] < 0) or (j == n - 1 or s[j + 1] < 0)
        out.append((int(s[i]), j - i + 1, bool(censored)))
        i = j + 1
    return out


def check_split_isolation(ts_by_split: Dict[str, np.ndarray],
                          bounds: Dict[str, Tuple[str, str]], window_hours: int = 24) -> Dict[str, object]:
    """Chronological isolation: each target lies in its split, its whole 24 h
    history lies in the same split, targets strictly increase, splits are ordered
    and disjoint.  Raises ValueError on violation."""
    info = {}
    prev_end = None
    for name in ("train", "validation", "test"):
        ts = np.asarray(ts_by_split[name]).astype("datetime64[ns]")
        start = np.datetime64(bounds[name][0], "ns")
        end_excl = np.datetime64(bounds[name][1], "ns") + np.timedelta64(1, "D")
        if np.any(np.diff(ts.astype(np.int64)) <= 0):
            raise ValueError(f"{name}: target timestamps not strictly increasing")
        if ts.min() < start + np.timedelta64(window_hours, "h"):
            raise ValueError(f"{name}: a window would reach before the split start")
        if ts.max() >= end_excl:
            raise ValueError(f"{name}: a target lies after the split end")
        if prev_end is not None and start < prev_end:
            raise ValueError(f"{name}: overlaps the previous split")
        prev_end = end_excl
        info[name] = {"first_target": str(ts.min()), "last_target": str(ts.max())}
    return info
