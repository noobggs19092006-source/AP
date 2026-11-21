"""
evaluation.py – Three-class / binary-change metrics, exact A_change / A_stable,
high-entry, Brier, LogLoss, ECE.
"""
import numpy as np
from sklearn.metrics import (confusion_matrix, f1_score, precision_recall_fscore_support,
                             balanced_accuracy_score)

EPS_LOGLOSS = 1e-12


def ensure_prob_simplex(probs):
    """Clip + renormalize probability rows to [eps, 1] then sum-to-1."""
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, EPS_LOGLOSS, None)
    s = probs.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return probs / s


def compute_metrics(y_true, y_pred, current_state, probs=None, scope_label=""):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    N = len(y_true)
    acc = float(np.mean(y_true == y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    prec_per, rec_per, f1_per, support_per = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    conf = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()

    classwise = [{"class": int(c), "precision": round(float(p), 8),
                   "recall": round(float(r), 8), "f1": round(float(f), 8),
                   "support": int(s)} for c, p, r, f, s in zip([0, 1, 2], prec_per, rec_per, f1_per, support_per)]

    # Scope A subset
    mask_a = np.asarray(current_state) >= 0
    y_a = y_true[mask_a]; pred_a = y_pred[mask_a]; cs_a = current_state[mask_a]
    Na = int(mask_a.sum())

    actual_change = (y_a != cs_a).astype(bool)
    predicted_change = (pred_a != cs_a).astype(bool)
    tp = int((actual_change & predicted_change).sum())
    fp = int((~actual_change & predicted_change).sum())
    fn = int((actual_change & ~predicted_change).sum())
    tn = int((~actual_change & ~predicted_change).sum())
    ch_prec = tp / (tp + fp) if (tp + fp) else 0.0
    ch_rec  = tp / (tp + fn) if (tp + fn) else 0.0
    ch_f1   = 2 * ch_prec * ch_rec / (ch_prec + ch_rec) if (ch_prec + ch_rec) else 0.0

    n_changes = int(actual_change.sum())
    correct_change = int((actual_change & (y_a == pred_a)).sum())
    A_change = correct_change / n_changes if n_changes else 0.0
    stable_mask = ~actual_change
    n_stable = int(stable_mask.sum())
    A_stable = float((y_a[stable_mask] == pred_a[stable_mask]).mean()) if n_stable else 0.0

    # High-entry (Scope A only)
    h_actual = ((cs_a != 2) & (y_a == 2)).astype(bool)
    h_pred   = ((cs_a != 2) & (pred_a == 2)).astype(bool)
    ht = int(h_actual.sum()); hp_ = int(h_pred.sum())
    h_tp = int((h_actual & h_pred).sum())
    h_fp = int((~h_actual & h_pred).sum())
    h_fn = int((h_actual & ~h_pred).sum())
    h_prec = h_tp / (h_tp + h_fp) if (h_tp + h_fp) else 0.0
    h_rec  = h_tp / (h_tp + h_fn) if (h_tp + h_fn) else 0.0
    h_f1   = 2 * h_prec * h_rec / (h_prec + h_rec) if (h_prec + h_rec) else 0.0

    prob_metrics = {}
    if probs is not None and probs.shape[0] == N:
        # Probability metrics are only meaningful on Scope A rows (observed current state)
        mask_a = np.asarray(current_state) >= 0
        if mask_a.all():
            pp = ensure_prob_simplex(probs)
            y_true_p = y_true
        else:
            pp = ensure_prob_simplex(probs[mask_a])
            y_true_p = y_true[mask_a]
        one_hot = np.zeros_like(pp)
        for k in range(3):
            one_hot[:, k] = (y_true_p == k).astype(float)
        brier = float(np.mean(np.sum((pp - one_hot) ** 2, axis=1)))
        true_probs = pp[np.arange(len(y_true_p)), y_true_p]
        logloss = float(-np.mean(np.log(np.maximum(true_probs, EPS_LOGLOSS))))
        prob_metrics = {"brier": round(brier, 8), "log_loss": round(logloss, 8)}

    return {
        "scope": scope_label,
        "N": N, "scopeA_N": Na,
        "accuracy": round(acc, 8),
        "macro_f1": round(macro_f1, 8),
        "confusion_matrix": conf,
        "classwise": classwise,
        "change_TP": tp, "change_FP": fp, "change_FN": fn, "change_TN": tn,
        "change_precision": round(ch_prec, 8),
        "change_recall": round(ch_rec, 8),
        "change_F1": round(ch_f1, 8),
        "A_change_numerator": correct_change,
        "A_change_denominator": n_changes,
        "A_change": round(A_change, 8),
        "A_stable_numerator": int((y_a[stable_mask] == pred_a[stable_mask]).sum()),
        "A_stable_denominator": n_stable,
        "A_stable": round(A_stable, 8),
        "high_entry_actual": ht,
        "high_entry_predicted": hp_,
        "high_TP": h_tp, "high_FP": h_fp, "high_FN": h_fn,
        "high_precision": round(h_prec, 8),
        "high_recall": round(h_rec, 8),
        "high_F1": round(h_f1, 8),
        **prob_metrics,
    }


def ece(probs, y_true, n_bins=10):
    """Expected Calibration Error with documented binning.
    Only valid (finite) probability rows are considered."""
    pp = np.asarray(probs, dtype=np.float64)
    if pp.ndim == 1:
        pp = pp.reshape(-1, 1)
    valid_mask = np.isfinite(pp).all(axis=1)
    pp = pp[valid_mask]
    y_true = np.asarray(y_true, dtype=int)[valid_mask]
    conf = np.max(pp, axis=1)
    correct = (np.argmax(pp, axis=1) == y_true).astype(float)
    N = len(conf)
    if N == 0:
        return 0.0, {}
    boundaries = np.linspace(0, 1, n_bins + 1)
    bin_data = {}
    ece_val = 0.0
    for i in range(n_bins):
        mask = (conf > boundaries[i]) & (conf <= boundaries[i + 1])
        n_bin = int(mask.sum())
        if n_bin > 0:
            acc = float(correct[mask].mean())
            conf_m = float(conf[mask].mean())
            ece_val += n_bin * abs(acc - conf_m)
            bin_data[i] = {"bin_start": round(float(boundaries[i]), 4),
                            "bin_end": round(float(boundaries[i + 1]), 4),
                            "count": n_bin,
                            "avg_confidence": round(conf_m, 4),
                            "empirical_accuracy": round(acc, 4)}
        else:
            bin_data[i] = {"bin_start": round(float(boundaries[i]), 4),
                            "bin_end": round(float(boundaries[i + 1]), 4),
                            "count": 0,
                            "avg_confidence": 0.0,
                            "empirical_accuracy": 0.0}
    return round(ece_val / N, 8), bin_data
