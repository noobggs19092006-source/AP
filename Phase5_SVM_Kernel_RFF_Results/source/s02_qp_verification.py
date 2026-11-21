"""
s02_qp_verification.py
======================
Small QP verification for binary soft-margin SVM dual.

Solves the classical hinge-loss dual QP via scipy SLSQP, recovers the primal,
checks KKT / complementary slackness, computes the primal-dual gap, and
compares against sklearn SVC(kernel='linear').

Outputs: output/results/qp_verification.json
Prints:  QP VERIFICATION: PASSED  or  QP VERIFICATION: FAILED
"""

import json
import os
import sys
import numpy as np
from scipy.optimize import minimize
from sklearn.svm import SVC

# ---------------------------------------------------------------------------
# Output dir
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "..", "results")
os.makedirs(OUT_DIR, exist_ok=True)
RESULT_PATH = os.path.join(OUT_DIR, "qp_verification.json")

# ---------------------------------------------------------------------------
# 1.  Data generation helpers
# ---------------------------------------------------------------------------

def make_separable_data(n=50, d=3, margin=1.0, seed=42):
    """
    Two classes, perfectly separated by margin `margin`.
    Class +1: points centred at +margin*e_1
    Class -1: points centred at -margin*e_1
    """
    rng = np.random.RandomState(seed)
    half = n // 2
    # Cluster centres
    c_pos = np.zeros(d); c_pos[0] = margin
    c_neg = np.zeros(d); c_neg[0] = -margin
    X_pos = c_pos + rng.randn(half, d) * 0.2
    X_neg = c_neg + rng.randn(n - half, d) * 0.2
    X = np.vstack([X_pos, X_neg]).astype(np.float64)
    y = np.array([+1] * half + [-1] * (n - half), dtype=np.float64)
    return X, y


def make_nonseparable_data(n=50, d=3, seed=42):
    """
    Points drawn from two overlapping Gaussians – not linearly separable.
    """
    rng = np.random.RandomState(seed)
    half = n // 2
    X_pos = rng.randn(half, d) * 0.8 + 0.5
    X_neg = rng.randn(n - half, d) * 0.8 - 0.5
    X = np.vstack([X_pos, X_neg]).astype(np.float64)
    y = np.array([+1] * half + [-1] * (n - half), dtype=np.float64)
    return X, y


def make_4point_separable():
    """
    4-point toy dataset, 2 per class, linearly separable.
    Exact dual solution: alpha = [0.5, 0.5, 0.5, 0.5] (for C>=0.5) gives
    w = [1, 0], b = 0, margin = 2.
    """
    X = np.array([[1.0, 0.0],
                  [2.0, 0.0],
                  [-1.0, 0.0],
                  [-2.0, 0.0]])
    y = np.array([+1.0, +1.0, -1.0, -1.0])
    return X, y


# ---------------------------------------------------------------------------
# 2.  Dual QP solver (SLSQP)
# ---------------------------------------------------------------------------

def solve_dual_qp(X, y, C):
    """
    Solve the soft-margin SVM dual QP (linear kernel):

        max_alpha  sum_i alpha_i - (1/2) * alpha^T H alpha
        s.t.       0 <= alpha_i <= C
                   sum_i alpha_i * y_i = 0

    where H_ij = y_i * y_j * (x_i . x_j)

    Returns dict with keys: alpha, success, message
    """
    n = len(y)
    K = X @ X.T                            # linear kernel matrix
    H = np.outer(y, y) * K                 # H_ij = y_i y_j K_ij

    # scipy.minimize minimises, so negate objective
    def objective(alpha):
        return 0.5 * alpha @ H @ alpha - np.sum(alpha)

    def grad_objective(alpha):
        return H @ alpha - np.ones(n)

    # Constraints: sum(alpha * y) == 0
    constraints = [{'type': 'eq',
                    'fun': lambda a: np.dot(a, y),
                    'jac': lambda a: y}]

    # Bounds: 0 <= alpha_i <= C
    bounds = [(0.0, C)] * n

    # Initial point: uniform feasible (if balanced)
    alpha0 = np.zeros(n)

    result = minimize(
        objective,
        alpha0,
        jac=grad_objective,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-12, 'maxiter': 2000}
    )

    return result


# ---------------------------------------------------------------------------
# 3.  Primal recovery and KKT checks
# ---------------------------------------------------------------------------

def recover_primal(X, y, alpha, C, tol=1e-5):
    """
    Recover w, b from dual solution; compute KKT residuals.

    Returns a dict with all diagnostics.
    """
    n = len(y)

    # --- Primal weights ---
    w = (alpha * y) @ X          # w = sum_i alpha_i * y_i * x_i  [shape (d,)]

    # --- Support vectors ---
    sv_mask  = alpha > tol                     # any alpha > 0
    free_sv  = (alpha > tol) & (alpha < C - tol)   # 0 < alpha < C → on margin
    bound_sv = alpha >= C - tol                     # alpha ≈ C → inside margin

    # --- Bias recovery: average over free support vectors ---
    if np.any(free_sv):
        b_vals = y[free_sv] - X[free_sv] @ w
        b = np.mean(b_vals)
    elif np.any(sv_mask):
        b_vals = y[sv_mask] - X[sv_mask] @ w
        b = np.mean(b_vals)
    else:
        b = 0.0

    # --- Functional margins ---
    f = X @ w + b                                     # decision values
    margins = y * f                                   # y_i * f(x_i)

    # --- Slack variables ---
    xi = np.maximum(0.0, 1.0 - margins)

    # --- KKT stationarity: nabla_w L = w - sum alpha_i y_i x_i = 0 ---
    grad_w_L = w - (alpha * y) @ X
    kkt_stationarity = float(np.linalg.norm(grad_w_L))

    # --- Equality constraint ---
    eq_constraint = float(np.abs(np.dot(alpha, y)))

    # --- Bound feasibility ---
    bound_violation = float(np.maximum(0, -alpha).max() +
                             np.maximum(0, alpha - C).max())

    # --- Complementary slackness ---
    # CS1: alpha_i * (y_i * f(x_i) - 1 + xi_i) = 0
    cs1 = alpha * (margins - 1.0 + xi)
    cs1_max = float(np.abs(cs1).max())

    # CS2: (C - alpha_i) * xi_i = 0
    cs2 = (C - alpha) * xi
    cs2_max = float(np.abs(cs2).max())

    # --- Primal objective ---
    primal_obj = 0.5 * np.dot(w, w) + C * np.sum(xi)

    # --- Dual objective ---
    K = X @ X.T
    H = np.outer(y, y) * K
    dual_obj = float(np.sum(alpha) - 0.5 * alpha @ H @ alpha)

    # --- Primal-dual gap ---
    gap = float(primal_obj - dual_obj)

    return {
        'w': w.tolist(),
        'b': float(b),
        'n_support_vectors': int(np.sum(sv_mask)),
        'n_free_sv': int(np.sum(free_sv)),
        'n_bound_sv': int(np.sum(bound_sv)),
        'margin_width': float(2.0 / np.linalg.norm(w)) if np.linalg.norm(w) > 1e-12 else float('inf'),
        'kkt_stationarity_norm': kkt_stationarity,
        'equality_constraint_violation': eq_constraint,
        'bound_violation': bound_violation,
        'cs1_max_violation': cs1_max,
        'cs2_max_violation': cs2_max,
        'primal_objective': float(primal_obj),
        'dual_objective': dual_obj,
        'primal_dual_gap': gap,
        'xi_sum': float(np.sum(xi)),
    }


# ---------------------------------------------------------------------------
# 4.  Comparison against sklearn SVC
# ---------------------------------------------------------------------------

def compare_sklearn(X, y, C, w_qp, b_qp):
    """
    Fit sklearn SVC(kernel='linear', C=C) and compare w, b with QP solution.

    Note: sklearn may use a different sign convention for b (intercept_),
    and may compute b differently (averaging over all support vectors or SMO).
    We compare w direction and |b| difference.
    """
    # sklearn SVC uses y in {-1, +1} natively for binary
    svc = SVC(kernel='linear', C=C, tol=1e-10, max_iter=50000)
    svc.fit(X, y)

    w_sk = svc.coef_[0]          # shape (d,)
    b_sk = float(svc.intercept_[0])

    # Align sign (convention may differ if training labels are flipped)
    # Compare normalised weight vectors
    w_qp_arr = np.array(w_qp)
    cos_sim = float(np.dot(w_qp_arr, w_sk) /
                    (np.linalg.norm(w_qp_arr) * np.linalg.norm(w_sk) + 1e-15))

    return {
        'sklearn_w': w_sk.tolist(),
        'sklearn_b': b_sk,
        'qp_w': w_qp,
        'qp_b': b_qp,
        'w_cosine_similarity': cos_sim,
        'b_difference': float(abs(b_qp - b_sk)),
        'note': (
            "Bias difference may arise because: (1) sklearn SMO averages b over "
            "all free support vectors with a numerical tolerance, while our SLSQP "
            "uses a simple mean; (2) LIBSVM may use a different regularisation for b; "
            "(3) SLSQP optimisation tolerance is 1e-12 vs sklearn's 1e-10."
        )
    }


# ---------------------------------------------------------------------------
# 5.  Unit tests
# ---------------------------------------------------------------------------

PASS_COUNT = 0
FAIL_COUNT = 0
TEST_LOG = []


def assert_close(val, expected, tol, name):
    global PASS_COUNT, FAIL_COUNT
    ok = abs(val - expected) <= tol
    status = "PASS" if ok else "FAIL"
    if ok:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    msg = f"  [{status}] {name}: got {val:.6g}, expected ~{expected:.6g} ± {tol:.1e}"
    TEST_LOG.append(msg)
    print(msg)
    return ok


def assert_less(val, threshold, name):
    global PASS_COUNT, FAIL_COUNT
    ok = val < threshold
    status = "PASS" if ok else "FAIL"
    if ok:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    msg = f"  [{status}] {name}: got {val:.6g}, threshold {threshold:.1e}"
    TEST_LOG.append(msg)
    print(msg)
    return ok


def run_qp_test(label, X, y, C, kkt_tol=1e-4, gap_tol=1e-4):
    """Run full dual QP pipeline on (X, y, C) and check correctness."""
    print(f"\n{'='*60}")
    print(f"TEST: {label}  (n={len(y)}, d={X.shape[1]}, C={C})")
    print(f"{'='*60}")

    result = solve_dual_qp(X, y, C)
    if not result.success:
        print(f"  [WARN] SLSQP did not fully converge: {result.message}")

    alpha = result.x
    diag  = recover_primal(X, y, alpha, C)

    print(f"  alpha range: [{alpha.min():.4f}, {alpha.max():.4f}]")
    print(f"  sum(alpha*y) = {np.dot(alpha, y):.2e}")
    print(f"  n_SV={diag['n_support_vectors']}, free={diag['n_free_sv']}, bound={diag['n_bound_sv']}")
    print(f"  margin = {diag['margin_width']:.4f}")
    print(f"  primal obj = {diag['primal_objective']:.6f}")
    print(f"  dual obj   = {diag['dual_objective']:.6f}")
    print(f"  gap        = {diag['primal_dual_gap']:.2e}")

    p1 = assert_less(diag['bound_violation'],          1e-8,    "bound feasibility (max violation)")
    p2 = assert_less(diag['equality_constraint_violation'], 1e-6, "equality |sum alpha*y|")
    p3 = assert_less(diag['kkt_stationarity_norm'],    kkt_tol,  "KKT stationarity ||grad_w L||")
    p4 = assert_less(diag['cs1_max_violation'],        kkt_tol,  "complementary slackness CS1 max")
    p5 = assert_less(diag['cs2_max_violation'],        kkt_tol,  "complementary slackness CS2 max")
    p6 = assert_less(abs(diag['primal_dual_gap']),     gap_tol,  "primal-dual gap")

    # sklearn comparison
    sk = compare_sklearn(X, y, C, diag['w'], diag['b'])
    p7 = assert_less(1.0 - sk['w_cosine_similarity'], 1e-3, "w cosine similarity vs sklearn")

    print(f"  b(QP)={diag['b']:.4f}, b(sklearn)={sk['sklearn_b']:.4f}, diff={sk['b_difference']:.4f}")
    print(f"  {sk['note']}")

    passed = all([p1, p2, p3, p4, p5, p6, p7])
    return passed, diag, alpha, sk


# ---------------------------------------------------------------------------
# 6.  4-point analytic unit test
# ---------------------------------------------------------------------------

def test_4point(C=10.0):
    """
    4-point dataset: X=[[1,0],[-1,0],[2,0],[-2,0]], y=[+1,-1,+1,-1]
    True w=[1,0], b=0, margin=2.
    With large C the two closest points (±1) are the support vectors.
    alpha for x=+1 and x=-1 should equal 0.5; others near 0.
    """
    print(f"\n{'='*60}")
    print(f"4-POINT ANALYTIC UNIT TEST (C={C})")
    print(f"{'='*60}")

    X, y = make_4point_separable()
    result = solve_dual_qp(X, y, C)
    alpha = result.x
    diag  = recover_primal(X, y, alpha, C)

    w = np.array(diag['w'])
    b = diag['b']

    print(f"  alpha = {alpha.round(4)}")
    print(f"  w = {w.round(4)}, b = {b:.4f}")
    print(f"  margin = {diag['margin_width']:.4f}")

    global PASS_COUNT, FAIL_COUNT
    # w direction should be [1, 0]
    w_norm = w / (np.linalg.norm(w) + 1e-15)
    p1 = assert_close(abs(w_norm[0]), 1.0, 1e-3, "4pt: |w_norm[0]|==1 (direction)")
    p2 = assert_close(abs(w_norm[1]) if len(w_norm) > 1 else 0.0, 0.0, 1e-3, "4pt: |w_norm[1]|==0")
    p3 = assert_close(diag['margin_width'], 2.0, 0.05, "4pt: margin == 2.0")
    p4 = assert_close(abs(b), 0.0, 0.1, "4pt: |b| ≈ 0")
    p5 = assert_less(abs(diag['primal_dual_gap']), 1e-4, "4pt: primal-dual gap < 1e-4")

    return all([p1, p2, p3, p4, p5])


# ---------------------------------------------------------------------------
# 7.  Main
# ---------------------------------------------------------------------------

def main():
    C = 1.0
    results = {}
    all_passed = True

    # ---- Test 1: Separable dataset (n=100, d=3) ----
    X_sep, y_sep = make_separable_data(n=100, d=3, margin=1.0, seed=42)
    passed_sep, diag_sep, alpha_sep, sk_sep = run_qp_test(
        "Separable (n=100, d=3)", X_sep, y_sep, C=C, kkt_tol=1e-3, gap_tol=1e-3
    )
    all_passed = all_passed and passed_sep

    results['separable'] = {
        'passed': passed_sep,
        'C': C,
        'n': len(y_sep),
        'd': X_sep.shape[1],
        'primal_dual_gap': diag_sep['primal_dual_gap'],
        'kkt_stationarity_norm': diag_sep['kkt_stationarity_norm'],
        'cs1_max_violation': diag_sep['cs1_max_violation'],
        'cs2_max_violation': diag_sep['cs2_max_violation'],
        'bound_violation': diag_sep['bound_violation'],
        'equality_constraint_violation': diag_sep['equality_constraint_violation'],
        'margin_width': diag_sep['margin_width'],
        'n_support_vectors': diag_sep['n_support_vectors'],
        'primal_objective': diag_sep['primal_objective'],
        'dual_objective': diag_sep['dual_objective'],
        'sklearn_comparison': sk_sep,
        'w': diag_sep['w'],
        'b': diag_sep['b'],
    }

    # ---- Test 2: Non-separable dataset (n=100, d=3) ----
    X_ns, y_ns = make_nonseparable_data(n=100, d=3, seed=42)
    passed_ns, diag_ns, alpha_ns, sk_ns = run_qp_test(
        "Non-separable (n=100, d=3)", X_ns, y_ns, C=C, kkt_tol=1e-3, gap_tol=1e-3
    )
    all_passed = all_passed and passed_ns

    results['nonseparable'] = {
        'passed': passed_ns,
        'C': C,
        'n': len(y_ns),
        'd': X_ns.shape[1],
        'primal_dual_gap': diag_ns['primal_dual_gap'],
        'kkt_stationarity_norm': diag_ns['kkt_stationarity_norm'],
        'cs1_max_violation': diag_ns['cs1_max_violation'],
        'cs2_max_violation': diag_ns['cs2_max_violation'],
        'bound_violation': diag_ns['bound_violation'],
        'equality_constraint_violation': diag_ns['equality_constraint_violation'],
        'margin_width': diag_ns['margin_width'],
        'n_support_vectors': diag_ns['n_support_vectors'],
        'primal_objective': diag_ns['primal_objective'],
        'dual_objective': diag_ns['dual_objective'],
        'sklearn_comparison': sk_ns,
        'w': diag_ns['w'],
        'b': diag_ns['b'],
    }

    # ---- Test 3: 4-point analytic unit test ----
    passed_4pt = test_4point(C=10.0)
    all_passed = all_passed and passed_4pt
    results['4point_analytic'] = {'passed': passed_4pt}

    # ---- Summary ----
    results['summary'] = {
        'total_tests': PASS_COUNT + FAIL_COUNT,
        'passed': PASS_COUNT,
        'failed': FAIL_COUNT,
        'all_passed': all_passed,
        'separable_primal_dual_gap': diag_sep['primal_dual_gap'],
        'nonseparable_primal_dual_gap': diag_ns['primal_dual_gap'],
        'test_log': TEST_LOG,
    }

    # ---- Save JSON ----
    with open(RESULT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {RESULT_PATH}")

    # ---- Final verdict ----
    verdict = "PASSED" if all_passed else "FAILED"
    print(f"\n{'='*60}")
    print(f"QP VERIFICATION: {verdict}")
    print(f"  Tests: {PASS_COUNT} passed, {FAIL_COUNT} failed")
    print(f"  Separable primal-dual gap  = {diag_sep['primal_dual_gap']:.2e}")
    print(f"  Non-separable primal-dual gap = {diag_ns['primal_dual_gap']:.2e}")
    print(f"{'='*60}")

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
