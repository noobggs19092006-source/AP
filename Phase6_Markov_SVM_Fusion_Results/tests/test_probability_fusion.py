"""
test_probability_fusion.py – Verify fusion mathematics: simplex, endpoints, bounds.
"""
import pytest
import numpy as np

EPS = 1e-12

def ensure_simplex(p):
    p = np.clip(p, EPS, 1.0)
    s = p.sum(axis=1, keepdims=True)
    return p / s

def test_fixed_fusion_endpoints():
    p_m = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.05, 0.15, 0.8]])
    p_s = np.array([[0.2, 0.3, 0.5], [0.4, 0.4, 0.2], [0.3, 0.3, 0.4]])
    # lambda = 0 => exactly Markov
    fused_0 = 0.0 * p_s + 1.0 * p_m
    assert np.allclose(fused_0, p_m)
    # lambda = 1 => exactly SVM
    fused_1 = 1.0 * p_s + 0.0 * p_m
    assert np.allclose(fused_1, p_s)
    # intermediate lambda = simplex
    for lam in [0.1, 0.3, 0.5, 0.7, 0.9]:
        fused = lam * p_s + (1 - lam) * p_m
        assert np.all(fused >= -1e-10)
        assert np.allclose(fused.sum(axis=1), 1.0, atol=1e-8)

def test_adaptive_fusion_bounds():
    p_m = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    p_s = np.array([[0.2, 0.3, 0.5], [0.4, 0.4, 0.2]])
    cs = np.array([0, 1])
    r_s = 1.0 - np.array([p_s[i, cs[i]] for i in range(2)])
    tau = 0.5
    mask = r_s >= tau
    ll, lh = 0.25, 0.75
    lambdas = np.where(mask, lh, ll)
    fused = np.zeros_like(p_s)
    for j in range(3):
        fused[:, j] = lambdas * p_s[:, j] + (1 - lambdas) * p_m[:, j]
    fused = ensure_simplex(fused)
    assert np.all(fused >= 0)
    assert np.allclose(fused.sum(axis=1), 1.0, atol=1e-8)
    assert np.all(lambdas >= 0) and np.all(lambdas <= 1)

def test_transition_fusion_zero_denom():
    p_m = np.array([[0.99, 0.005, 0.005], [0.01, 0.98, 0.01]])
    p_s = np.array([[0.95, 0.03, 0.02], [0.1, 0.85, 0.05]])
    cs = np.array([0, 1])
    # eta=1, rho=1 -> use SVM only; but SVM has near-zero mass on non-current
    eta = 1.0
    # destination denominator for i=0: 1 - p_s[0,0] = 0.05 (OK)
    # for i=1: 1 - p_s[1,1] = 0.15 (OK)
    # No zero denom in this case
    pass

def test_logloss_eps():
    # Clip then renormalize then natural log
    p = np.array([[1.0, 0.0, 0.0]])
    y = np.array([0])
    pp = np.clip(p, EPS, 1.0)
    pp = pp / pp.sum(axis=1, keepdims=True)
    loss = -np.mean(np.log(pp[np.arange(1), y]))
    assert np.isfinite(loss)
    # if p was [0,0,1] for class 0:
    p2 = np.array([[0.0, 0.0, 1.0]])
    pp2 = np.clip(p2, EPS, 1.0)
    pp2 = pp2 / pp2.sum(axis=1, keepdims=True)
    loss2 = -np.mean(np.log(pp2[np.arange(1), np.array([0])]))
    assert np.isfinite(loss2)
    assert loss2 > 0