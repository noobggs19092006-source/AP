#!/usr/bin/env python3
"""
test_phase5_full.py - the 14 minimum tests of spec section 15, rewritten to be portable and to exercise the
REAL code (common.py, saved artifacts) rather than re-implementing logic inline.

Run:  python -m pytest tests -v --input-dir <phase2-4 inputs> --output-dir <package root>
Tests needing the large input files are skipped (not failed) when --input-dir is not supplied.
"""
import hashlib
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import LinearSVC

from common import (apply_frozen_scaler, load_frozen_scaler, prob_metrics, transition_metrics)

CONFIGS = ["pca_A", "pca_B", "pca_C", "hosvd_A", "hosvd_B", "hosvd_C", "hosvd_B_alt1", "hosvd_B_alt2"]
DIMS = {"pca_A": 48, "pca_B": 192, "pca_C": 576, "hosvd_A": 48, "hosvd_B": 192, "hosvd_C": 576, "hosvd_B_alt1": 192, "hosvd_B_alt2": 192}
NS = {"train": 24108, "validation": 4239, "test": 5720}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def resolve_feature(in_dir: Path, rel: str) -> Path:
    """Exact name, else a UNIQUE '(1)'-suffixed upload copy (never ambiguous)."""
    p = in_dir / rel
    if p.exists():
        return p
    cands = list(in_dir.glob(Path(rel).stem + " (*)" + Path(rel).suffix))
    assert len(cands) == 1, f"cannot resolve {rel} uniquely: {cands}"
    return cands[0]


# 1 ----------------------------------------------------------------------------------------------
def test_01_npz_five_key_schema(in_dir):
    for cfg in CONFIGS:
        for sp, n in NS.items():
            with np.load(resolve_feature(in_dir, f"features_{sp}_{cfg}.npz")) as d:
                assert sorted(d.files) == ["F", "current_pm25", "current_state", "target_timestamp", "y"]
                assert d["F"].shape == (n, DIMS[cfg]) and d["F"].dtype == np.float32 and np.isfinite(d["F"]).all()
                assert d["y"].dtype == np.int8 and set(np.unique(d["y"])) <= {0, 1, 2}
                assert d["current_state"].dtype == np.int8 and d["current_pm25"].dtype == np.float32
                assert d["target_timestamp"].dtype == np.dtype("datetime64[ns]")
                # unknown current state <=> NaN current_pm25
                assert np.array_equal(d["current_state"] == -1, np.isnan(d["current_pm25"]))
    # alignment with Phase 2 windows (metadata keys only; X is never loaded)
    for sp in NS:
        w = in_dir / f"{sp}_windows.npz"
        if not w.exists():
            pytest.skip(f"{w.name} not supplied")
        with np.load(w) as wz, np.load(in_dir / f"features_{sp}_pca_B.npz") as fz:
            assert np.array_equal(wz["y"], fz["y"]) and np.array_equal(wz["target_timestamp"], fz["target_timestamp"])
            assert np.array_equal(wz["current_state"], fz["current_state"])
            assert np.array_equal(wz["current_pm25"], fz["current_pm25"], equal_nan=True)


# 2 ----------------------------------------------------------------------------------------------
def test_02_manifest_sha256_integrity(in_dir):
    man = pd.read_csv(in_dir / "phase4_feature_manifest.csv")
    assert len(man) == 24
    for _, r in man.iterrows():
        p = resolve_feature(in_dir, r["relative_path"])
        assert p.stat().st_size == r["actual_file_bytes"]
        assert sha256(p) == r["sha256"]


def test_02b_duplicate_filename_resolution(tmp_path):
    (tmp_path / "a (1).npz").write_bytes(b"x")
    assert resolve_feature(tmp_path, "a.npz").name == "a (1).npz"
    (tmp_path / "a (2).npz").write_bytes(b"y")
    with pytest.raises(AssertionError):
        resolve_feature(tmp_path, "a.npz")          # ambiguous -> rejected


# 3 ----------------------------------------------------------------------------------------------
def test_03_scalers_are_train_only(in_dir, out_dir):
    for cfg in CONFIGS:
        mean, scale = load_frozen_scaler(out_dir / "models" / f"scaler_{cfg}.npz")
        with np.load(in_dir / f"features_train_{cfg}.npz") as tr:
            F = tr["F"].astype(np.float64)
        assert np.allclose(mean, F.mean(0), atol=1e-6), cfg
        sd = F.std(0)
        assert np.allclose(scale, np.where(sd == 0, 1.0, sd), rtol=1e-5, atol=1e-8), cfg
        with np.load(out_dir / "models" / f"scaler_{cfg}.npz") as z:
            assert int(z["n_samples_seen"]) == 24108
        with np.load(in_dir / f"features_validation_{cfg}.npz") as va:
            assert not np.allclose(mean, va["F"].astype(np.float64).mean(0), atol=1e-6)   # not fitted on validation


def test_03b_no_fit_on_validation_or_test_in_source():
    pat = re.compile(r"\.fit(_transform)?\(\s*[A-Za-z_]*(val|va\b|test|te\b|Xva|Xte|Zva|Zte|F_va|F_te)", re.I)
    bad = []
    for f in (Path(__file__).resolve().parent.parent / "source").glob("*.py"):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if pat.search(line) and not line.strip().startswith("#"):
                bad.append(f"{f.name}:{i}: {line.strip()}")
    assert not bad, bad


# 4 ----------------------------------------------------------------------------------------------
def test_04_deterministic_configurations(need_results):
    rng = np.random.RandomState(42)
    X, y = rng.randn(120, 10), rng.choice([0, 1, 2], size=120)
    c1 = LinearSVC(C=1.0, dual=False, max_iter=2000, random_state=42).fit(X, y)
    c2 = LinearSVC(C=1.0, dual=False, max_iter=2000, random_state=42).fit(X, y)
    assert np.array_equal(c1.coef_, c2.coef_)
    out = need_results("results/stage_c_full_experiment.csv")
    d = pd.read_csv(out / "results/stage_c_full_experiment.csv")
    assert len(d) == 30 and not d.duplicated(["config", "D", "seed"]).any()
    assert sorted(d["seed"].unique()) == [42, 43, 44, 45, 46] and sorted(d["D"].unique()) == [256, 1024, 2048]


# 5 ----------------------------------------------------------------------------------------------
def test_05_hand_computed_rbf_kernel():
    gamma = 0.25
    x, z = np.array([[1., 2., 3.], [4., 5., 6.]]), np.array([[1.5, 2.5, 3.5], [0., 1., 2.]])
    assert abs(np.exp(-gamma * 0.75) - rbf_kernel(x, z, gamma=gamma)[0, 0]) < 1e-14
    hand = np.exp(-gamma * ((x[1] - z[1]) ** 2).sum())
    assert abs(hand - rbf_kernel(x, z, gamma=gamma)[1, 1]) < 1e-14


# 6 ----------------------------------------------------------------------------------------------
def test_06_rff_convention_reproducibility_and_monte_carlo():
    gamma, d, D = 0.3, 6, 8192
    X = np.random.RandomState(0).randn(40, d)
    s1, s2 = RBFSampler(gamma=gamma, n_components=D, random_state=7), RBFSampler(gamma=gamma, n_components=D, random_state=7)
    Z1, Z2 = s1.fit_transform(X), s2.fit_transform(X)
    assert np.array_equal(Z1, Z2)
    # manual sampler: omega ~ N(0, 2 gamma I), b ~ U(0, 2 pi), z = sqrt(2/D) cos(W^T x + b)
    rs = np.random.RandomState(7)
    W = np.sqrt(2 * gamma) * rs.normal(size=(d, D))
    b = rs.uniform(0, 2 * np.pi, size=D)
    Zm = np.sqrt(2.0 / D) * np.cos(X @ W + b)
    assert np.allclose(Zm, Z1, atol=1e-12), "RBFSampler gamma/normalisation convention differs from the derivation"
    assert abs(s1.random_weights_.std() - np.sqrt(2 * gamma)) < 0.02
    assert 0 <= s1.random_offset_.min() and s1.random_offset_.max() <= 2 * np.pi
    # Monte Carlo sanity: averaged over seeds the error shrinks with D (individual seeds may not be monotone)
    K = rbf_kernel(X, gamma=gamma)
    err = {}
    for D_ in (64, 1024, 16384):
        e = [np.linalg.norm(K - (lambda Z: Z @ Z.T)(RBFSampler(gamma=gamma, n_components=D_, random_state=s).fit_transform(X)))
             / np.linalg.norm(K) for s in range(5)]
        err[D_] = np.mean(e)
    assert err[64] > err[1024] > err[16384]
    assert err[16384] < 0.05


# 7 ----------------------------------------------------------------------------------------------
def test_07_qp_feasibility_and_kkt_direct():
    from s02_qp_verification import make_nonseparable_data, recover_primal, solve_dual_qp
    X, y = make_nonseparable_data(n=60, d=3, seed=1)
    res = solve_dual_qp(X, y, 1.0)
    diag = recover_primal(X, y, res.x, 1.0)
    assert diag["bound_violation"] < 1e-8 and diag["equality_constraint_violation"] < 1e-6
    assert diag["kkt_stationarity_norm"] < 1e-3


def test_07b_saved_qp_results(need_results):
    out = need_results("results/qp_verification.json")
    s = json.loads((out / "results/qp_verification.json").read_text())["summary"]
    assert s["all_passed"] and s["separable_primal_dual_gap"] < 1e-3 and s["nonseparable_primal_dual_gap"] < 1e-3


# 8 ----------------------------------------------------------------------------------------------
def test_08_class_order_and_shapes(out_dir):
    files = sorted((out_dir / "predictions").rglob("final_*.csv"))
    if not files:
        pytest.skip("no prediction CSVs")
    for pf in files:
        df = pd.read_csv(pf)
        for col in ["score_0", "score_1", "score_2", "prob_0", "prob_1", "prob_2", "hard_pred", "y", "current_state", "scope_a_eligible"]:
            assert col in df.columns, (pf.name, col)
        P = df[["prob_0", "prob_1", "prob_2"]].values
        if np.isfinite(P).all():
            assert (P >= -1e-9).all() and np.allclose(P.sum(1), 1, atol=1e-6), pf.name
        else:
            assert np.isnan(P).all(), f"{pf.name}: partially-missing probabilities"
        assert set(np.unique(df["hard_pred"])) <= {0, 1, 2}
    for pk in ("final_linear_pca_B.pkl", "calibrator_linear_pca_B.pkl"):
        if (out_dir / "models" / pk).exists():
            with open(out_dir / "models" / pk, "rb") as f:
                assert list(pickle.load(f).classes_) == [0, 1, 2]


# 9 ----------------------------------------------------------------------------------------------
def test_09_brier_and_logloss_known_examples():
    r = prob_metrics(np.array([0]), np.full((1, 3), 1 / 3))
    assert abs(r["brier"] - 2 / 3) < 1e-12 and abs(r["log_loss"] - np.log(3)) < 1e-12
    r = prob_metrics(np.array([1, 2]), np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]))      # perfect / confidently wrong
    assert abs(r["brier"] - (0 + 2.0) / 2) < 1e-9
    assert abs(r["log_loss"] - (0 + (-np.log(1e-12 / (1 + 1e-12)))) / 2) < 1e-6            # clipped at EPS=1e-12 + renormalised
    with pytest.raises(ValueError):
        prob_metrics(np.array([0]), np.ones((1, 2)))


# 10 ---------------------------------------------------------------------------------------------
def test_10_transitions_high_entry_and_edge_cases():
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    c = np.zeros(10, dtype=int)
    zero = transition_metrics(y, np.zeros(10, dtype=int), c)                        # never predicts a change
    assert (zero["trans_tp"], zero["trans_fp"], zero["trans_fn"], zero["trans_tn"]) == (0, 0, 6, 4)
    assert zero["trans_precision"] == zero["trans_recall"] == zero["trans_f1"] == 0.0
    assert zero["high_tp"] == 0 and zero["high_fn"] == 3 and zero["true_high_entries"] == 3
    # destination is distinguished from "predicts a change"
    p = np.array([0, 2, 2, 0, 1, 1, 0, 1, 0, 1])
    m = transition_metrics(y, p, c)
    # hand count: actual changes at rows 1,2,4,5,7,8; predicted changes at rows 1,2,4,5,7,9 -> TP=5 FP=1 FN=1
    assert (m["trans_tp"], m["trans_fp"], m["trans_fn"]) == (5, 1, 1)
    # correct destination only at rows 2 (2), 4 (1), 7 (1) -> 3 of the 6 actual changes
    assert m["trans_dest_accuracy"] == pytest.approx(3 / 6)
    # Scope A eligibility: rows with c = -1 are excluded entirely
    c2 = c.copy(); c2[[1, 2]] = -1
    m2 = transition_metrics(y, p, c2)
    assert m2["n_scope_a"] == 8 and m2["true_transitions"] == 4
    # persistence has zero predicted changes by construction
    assert transition_metrics(y, c, c)["pred_transitions"] == 0


# 11 ---------------------------------------------------------------------------------------------
def test_11_phase3_counts_and_alignment(in_dir, out_dir):
    v, t = (pd.read_csv(in_dir / f"markov_predictions_{s}.csv") for s in ("validation", "test"))
    va, ta = v[v.scope_a_eligible == True], t[t.scope_a_eligible == True]
    assert (len(va), len(ta)) == (4197, 5685)
    assert (int((va.y != va.current_state).sum()), int((ta.y != ta.current_state).sum())) == (488, 526)
    assert (int(((va.current_state != 2) & (va.y == 2)).sum()), int(((ta.current_state != 2) & (ta.y == 2)).sum())) == (122, 154)
    assert int((ta.pred_markov_scope_b != ta.current_state).sum()) == 0
    for f in sorted((out_dir / "predictions").rglob("final_*_test.csv")):
        d = pd.read_csv(f)
        assert len(d) == len(t) and np.array_equal(d.y.values, t.y.values) and np.array_equal(d.current_state.values, t.current_state.values)
        assert (pd.to_datetime(d.target_timestamp).values == pd.to_datetime(t.target_timestamp).values).all()


# 12 ---------------------------------------------------------------------------------------------
def test_12_selection_lock_before_test(out_dir):
    lock = out_dir / "manifests" / "LOCKED_SELECTION.json"
    if not lock.exists():
        pytest.skip("no lock manifest")
    L = json.loads(lock.read_text())
    assert L["test_metrics_used"] is False and "timestamp_utc" in L and "linear_pca_B" in L["selected_models"]
    src = (out_dir / "source" / "s06_selection_and_evaluation.py").read_text()
    assert src.index("LOCKED_SELECTION.json") < src.index('features_test_{cfg}.npz'), "lock must be written before test features are opened"
    for name in ("select_with_tie_rule", "select_rff_policy"):                       # selection helpers must not see test data
        body = (out_dir / "source" / "common.py").read_text().split(f"def {name}")[1].split("\ndef ")[0]
        assert "test" not in body.lower().replace("tie", "")
    rev = out_dir / "manifests" / "REVISED_SELECTION_post_audit.json"
    if rev.exists():
        R = json.loads(rev.read_text())
        assert R["status"] == "POST_AUDIT_REVISION" and R["test_metrics_used_for_selection"] is False and "provenance_caveat" in R


# 13 ---------------------------------------------------------------------------------------------
def test_13_reload_reproduces_predictions(in_dir, out_dir):
    mp = out_dir / "models" / "final_linear_pca_B.pkl"
    if not mp.exists():
        pytest.skip("model not present")
    mean, scale = load_frozen_scaler(out_dir / "models" / "scaler_pca_B.npz")
    with np.load(in_dir / "features_test_pca_B.npz") as d:
        X = apply_frozen_scaler(d["F"], mean, scale)
    clf = pickle.load(open(mp, "rb"))
    saved = pd.read_csv(out_dir / "predictions" / "final_linear_pca_B_test.csv")
    assert np.array_equal(clf.predict(X), saved.hard_pred.values)
    assert np.abs(clf.decision_function(X) - saved[["score_0", "score_1", "score_2"]].values).max() < 1e-9
    cal = pickle.load(open(out_dir / "models" / "calibrator_linear_pca_B.pkl", "rb"))
    assert np.abs(cal.predict_proba(clf.decision_function(X)) - saved[["prob_0", "prob_1", "prob_2"]].values).max() < 1e-9


def test_13b_reload_revised_rff(in_dir, out_dir):
    mp = out_dir / "models" / "final_revised_rff_pca_B.pkl"
    if not mp.exists():
        pytest.skip("revised RFF model not present")
    mean, scale = load_frozen_scaler(out_dir / "models" / "scaler_pca_B.npz")
    bundle = pickle.load(open(mp, "rb"))
    cal = pickle.load(open(out_dir / "models" / "calibrator_revised_rff_pca_B.pkl", "rb"))
    saved = pd.read_csv(out_dir / "predictions" / "revised" / "final_revised_rff_pca_B_test.csv")
    with np.load(in_dir / "features_test_pca_B.npz") as d:
        X = apply_frozen_scaler(d["F"][:800], mean, scale)
    Z = bundle["sampler"].transform(X)
    sc = bundle["classifier"].decision_function(Z)
    assert np.array_equal(bundle["classifier"].predict(Z), saved.hard_pred.values[:800])
    assert np.abs(sc - saved[["score_0", "score_1", "score_2"]].values[:800]).max() < 1e-9
    assert np.abs(cal.predict_proba(sc) - saved[["prob_0", "prob_1", "prob_2"]].values[:800]).max() < 1e-9
    # stored mapping file reproduces the sampler exactly
    mf = next((out_dir / "models").glob("rff_mapping_revised_pca_B_*.npz"))
    with np.load(mf) as z:
        assert np.array_equal(z["random_weights"], bundle["sampler"].random_weights_)


# 14 ---------------------------------------------------------------------------------------------
def test_14_figures_and_inventory(out_dir):
    figs = ["stage_a_c_curves.png", "stage_a_confusion_matrices.png", "rff_kernel_approx.png", "rff_macro_f1_vs_D.png",
            "markov_vs_svm_transitions.png", "high_entry_comparison.png", "pca_vs_hosvd_macro_f1_vs_dim.png",
            "hosvd_rank_comparison.png", "reconstruction_vs_classification.png", "pareto_performance_vs_storage.png",
            "pareto_performance_vs_runtime.png", "stage_b_rbf_heatmap_all.png"]
    for f in figs:
        assert (out_dir / "figures" / f).exists() and (out_dir / "figures" / f).stat().st_size > 5000, f
    inv = out_dir / "manifests" / "artifacts_inventory.json"
    if inv.exists():
        for e in json.loads(inv.read_text()):
            assert (out_dir / e["path"]).exists(), f"inventory lists missing file {e['path']}"
