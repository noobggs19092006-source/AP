"""
test_qp.py
==========
Unit tests for the QP verification pipeline (separable and non-separable cases).
Can be run directly with: python test_qp.py
or with:                  python -m pytest test_qp.py -v
"""

import sys
import os
import numpy as np
import pytest

# Allow imports from the source directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'source'))
from s02_qp_verification import (
    make_separable_data,
    make_nonseparable_data,
    make_4point_separable,
    solve_dual_qp,
    recover_primal,
    compare_sklearn,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
C = 1.0
KKT_TOL  = 1e-3
GAP_TOL  = 1e-3
BOUND_TOL = 1e-8
EQ_TOL   = 1e-6


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run_pipeline(X, y, C=C):
    result = solve_dual_qp(X, y, C)
    alpha  = result.x
    diag   = recover_primal(X, y, alpha, C)
    return alpha, diag


# ---------------------------------------------------------------------------
# Tests: Separable dataset
# ---------------------------------------------------------------------------

class TestSeparable:

    def setup_method(self):
        self.X, self.y = make_separable_data(n=100, d=3, margin=1.0, seed=42)
        self.alpha, self.diag = _run_pipeline(self.X, self.y, C=C)

    def test_bound_feasibility(self):
        """0 <= alpha_i <= C for all i."""
        assert self.diag['bound_violation'] < BOUND_TOL, (
            f"Bound violation: {self.diag['bound_violation']:.2e}"
        )

    def test_equality_constraint(self):
        """|sum alpha_i * y_i| < 1e-6."""
        assert self.diag['equality_constraint_violation'] < EQ_TOL, (
            f"Equality violation: {self.diag['equality_constraint_violation']:.2e}"
        )

    def test_kkt_stationarity(self):
        """||nabla_w L|| < KKT_TOL."""
        assert self.diag['kkt_stationarity_norm'] < KKT_TOL, (
            f"KKT stationarity: {self.diag['kkt_stationarity_norm']:.2e}"
        )

    def test_complementary_slackness_cs1(self):
        """alpha_i * (y_i * f(x_i) - 1 + xi_i) ≈ 0."""
        assert self.diag['cs1_max_violation'] < KKT_TOL, (
            f"CS1 violation: {self.diag['cs1_max_violation']:.2e}"
        )

    def test_complementary_slackness_cs2(self):
        """(C - alpha_i) * xi_i ≈ 0."""
        assert self.diag['cs2_max_violation'] < KKT_TOL, (
            f"CS2 violation: {self.diag['cs2_max_violation']:.2e}"
        )

    def test_primal_dual_gap(self):
        """Primal-dual gap < GAP_TOL (strong duality)."""
        assert abs(self.diag['primal_dual_gap']) < GAP_TOL, (
            f"Primal-dual gap: {self.diag['primal_dual_gap']:.2e}"
        )

    def test_primal_dual_ordering(self):
        """Dual obj <= Primal obj (weak duality). Gap >= -tolerance."""
        assert self.diag['primal_dual_gap'] >= -GAP_TOL, (
            f"Weak duality violated: gap = {self.diag['primal_dual_gap']:.2e}"
        )

    def test_has_support_vectors(self):
        """At least 1 support vector exists."""
        assert self.diag['n_support_vectors'] >= 1

    def test_sklearn_w_alignment(self):
        """QP and sklearn weight vectors are nearly co-linear (cosine > 0.999)."""
        sk = compare_sklearn(self.X, self.y, C, self.diag['w'], self.diag['b'])
        assert sk['w_cosine_similarity'] > 0.999, (
            f"Cosine similarity: {sk['w_cosine_similarity']:.4f}"
        )


# ---------------------------------------------------------------------------
# Tests: Non-separable dataset
# ---------------------------------------------------------------------------

class TestNonseparable:

    def setup_method(self):
        self.X, self.y = make_nonseparable_data(n=100, d=3, seed=42)
        self.alpha, self.diag = _run_pipeline(self.X, self.y, C=C)

    def test_bound_feasibility(self):
        assert self.diag['bound_violation'] < BOUND_TOL, (
            f"Bound violation: {self.diag['bound_violation']:.2e}"
        )

    def test_equality_constraint(self):
        assert self.diag['equality_constraint_violation'] < EQ_TOL, (
            f"Equality violation: {self.diag['equality_constraint_violation']:.2e}"
        )

    def test_kkt_stationarity(self):
        assert self.diag['kkt_stationarity_norm'] < KKT_TOL, (
            f"KKT stationarity: {self.diag['kkt_stationarity_norm']:.2e}"
        )

    def test_complementary_slackness_cs1(self):
        assert self.diag['cs1_max_violation'] < KKT_TOL, (
            f"CS1 violation: {self.diag['cs1_max_violation']:.2e}"
        )

    def test_complementary_slackness_cs2(self):
        assert self.diag['cs2_max_violation'] < KKT_TOL, (
            f"CS2 violation: {self.diag['cs2_max_violation']:.2e}"
        )

    def test_primal_dual_gap(self):
        assert abs(self.diag['primal_dual_gap']) < GAP_TOL, (
            f"Primal-dual gap: {self.diag['primal_dual_gap']:.2e}"
        )

    def test_has_bound_support_vectors(self):
        """Non-separable data must have some alpha_i = C (points inside margin)."""
        assert self.diag['n_bound_sv'] >= 1, (
            "Expected bound SVs (alpha==C) for non-separable data"
        )

    def test_nonzero_slack(self):
        """Non-separable: total slack > 0."""
        assert self.diag['xi_sum'] > 0.0, (
            f"Expected xi_sum > 0, got {self.diag['xi_sum']:.4f}"
        )

    def test_sklearn_w_alignment(self):
        sk = compare_sklearn(self.X, self.y, C, self.diag['w'], self.diag['b'])
        assert sk['w_cosine_similarity'] > 0.99, (
            f"Cosine similarity: {sk['w_cosine_similarity']:.4f}"
        )


# ---------------------------------------------------------------------------
# Tests: 4-point analytic dataset
# ---------------------------------------------------------------------------

class Test4PointAnalytic:

    def setup_method(self):
        self.X, self.y = make_4point_separable()
        self.C = 10.0
        self.alpha, self.diag = _run_pipeline(self.X, self.y, C=self.C)

    def test_weight_direction(self):
        """
        True w = [1, 0].  Normalised w must point in that direction.
        """
        w = np.array(self.diag['w'])
        w_norm = w / (np.linalg.norm(w) + 1e-15)
        # Accept either sign (depends on label ordering in sklearn/SLSQP)
        assert abs(abs(w_norm[0]) - 1.0) < 0.02, (
            f"|w_norm[0]| = {abs(w_norm[0]):.4f}, expected 1.0"
        )

    def test_margin(self):
        """Margin should be ~2.0 for 4-point dataset."""
        assert abs(self.diag['margin_width'] - 2.0) < 0.1, (
            f"margin = {self.diag['margin_width']:.4f}, expected ~2.0"
        )

    def test_bias_near_zero(self):
        """By symmetry b ≈ 0."""
        assert abs(self.diag['b']) < 0.15, (
            f"b = {self.diag['b']:.4f}, expected ~0"
        )

    def test_primal_dual_gap(self):
        assert abs(self.diag['primal_dual_gap']) < GAP_TOL, (
            f"Gap = {self.diag['primal_dual_gap']:.2e}"
        )

    def test_exactly_two_support_vectors(self):
        """
        The two nearest points (at distance 1 from origin) should be SVs.
        With large C, points at distance 2 have alpha ≈ 0.
        """
        assert self.diag['n_support_vectors'] == 2, (
            f"Expected 2 SVs, got {self.diag['n_support_vectors']}"
        )

    def test_kkt_stationarity(self):
        assert self.diag['kkt_stationarity_norm'] < KKT_TOL


# ---------------------------------------------------------------------------
# Run as script
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import subprocess
    ret = subprocess.run(
        [sys.executable, '-m', 'pytest', __file__, '-v', '--tb=short'],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    sys.exit(ret.returncode)
