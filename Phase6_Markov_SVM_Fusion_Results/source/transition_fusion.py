"""
transition_fusion.py – F3: change-likelihood q + conditional destination d.
Reconstruct p_F so that it sums to 1 and lies in [0,1].
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from evaluation import compute_metrics


def blended_change_probability(r_svm, r_markov, rho):
    return rho * r_svm + (1.0 - rho) * r_markov


def conditional_destination(p_svm, p_markov, eta, current_state):
    """For each sample, blend the conditional destination distributions from
    SVM and Markov.  For current state i, destination over j != i:
        d(j) = [eta*p_svm(j) + (1-eta)*p_markov(j)]
               / [eta*sum_{k!=i}p_svm(k) + (1-eta)*sum_{k!=i}p_markov(k)]
    Zero-denominator fallback: uniform over the K-1 non-current states.
    Returns (N,3) d matrix and count of fallbacks applied."""
    N, K = p_svm.shape
    d = np.zeros((N, K), dtype=np.float64)
    denom_count = 0
    for i in range(K):
        mask_i = current_state == i
        if not mask_i.any():
            continue
        # blended numerator for each destination j != i
        num = eta * p_svm + (1.0 - eta) * p_markov
        # blended denominator: sum over k != i
        denom_s = 1.0 - p_svm[:, i]
        denom_m = 1.0 - p_markov[:, i]
        denom = eta * denom_s + (1.0 - eta) * denom_m
        zero_mask = denom < 1e-12
        denom_count += int(zero_mask.sum())
        safe_denom = np.where(zero_mask, 1.0, denom)
        for j in range(K):
            if j == i:
                continue
            d[mask_i, j] = np.where(zero_mask[mask_i], 1.0 / (K - 1),
                                     num[mask_i, j] / safe_denom[mask_i])
    return d, denom_count


def reconstruct_from_q_d(q, d, current_state):
    """Reconstruct valid 3-class probabilities:
        p_F,t(i) = 1 - q_t
        p_F,t(j) = q_t * d_F,t(j)   for j != i
    q may be 1D (length N) or 2D (N,1)."""
    q = np.asarray(q, dtype=np.float64).ravel()
    N = q.shape[0]
    K = d.shape[1]
    p_out = np.zeros((N, K), dtype=np.float64)
    for i in range(N):
        ci = int(current_state[i])
        if 0 <= ci < K:
            rest = q[i]
            if rest < 0:
                rest = 0.0
            p_out[i, ci] = 1.0 - rest
            for j in range(K):
                if j != ci:
                    p_out[i, j] = rest * d[i, j]
        else:
            p_out[i] = 1.0 / K
    s = p_out.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return p_out / s


def fit_binary_change_clf(X_train, y_train_cs, y_train_y):
    """Train-only binary change classifier using features + current state.
    Returns (clf, scaler, q_calib)."""
    C = (y_train_y != y_train_cs).astype(int)
    h_train = np.hstack([X_train, y_train_cs.reshape(-1, 1).astype(float)])
    scaler = StandardScaler().fit(h_train)
    h_train_s = scaler.transform(h_train)
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver="lbfgs")
    clf.fit(h_train_s, C)
    return clf, scaler


def run_transition_fusion(p_markov, p_svm, y_true, current_state, rho_grid, eta_grid,
                           use_binary_classifier=False, X_train=None, y_train_cs=None, y_train_y=None):
    rows = []
    r_S = 1.0 - np.take_along_axis(p_svm, current_state[:, None].astype(int), axis=1).squeeze(-1)
    r_M = 1.0 - np.take_along_axis(p_markov, current_state[:, None].astype(int), axis=1).squeeze(-1)
    q_bin = None
    if use_binary_classifier and X_train is not None:
        clf, scaler = fit_binary_change_clf(X_train, y_train_cs, y_train_y)
        h = np.hstack([X_train, y_train_cs.reshape(-1, 1).astype(float)])
        h_s = scaler.transform(h)
        q_bin = np.clip(clf.predict_proba(h_s)[:, 1], 1e-6, 1 - 1e-6)
    for rho in rho_grid:
        for eta in eta_grid:
            if q_bin is not None:
                q_t = rho * q_bin + (1.0 - rho) * blended_change_probability(r_S, r_M, 0.5)
            else:
                q_t = blended_change_probability(r_S, r_M, rho)
            d_t, denom_count = conditional_destination(p_svm, p_markov, eta, current_state)
            p_fused = reconstruct_from_q_d(q_t, d_t, current_state)
            preds = np.argmax(p_fused, axis=1)
            met = compute_metrics(y_true, preds, current_state, probs=p_fused)
            met.update({
                "model": f"transition_rho{rho}_eta{eta}" + ("_binary" if q_bin is not None else ""),
                "fusion_type": "transition",
                "rho": rho, "eta": eta,
                "zero_denom_fallback_count": denom_count,
                "scope": "val",
                "q_mean": float(q_t.mean()),
                "q_std": float(q_t.std()),
            })
            rows.append(met)
    return rows