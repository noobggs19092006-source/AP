#!/usr/bin/env python3
"""
test_remediation.py - regression tests added for the audit remediation.

Covers: scaler-key handling (the Stage B 'mean_' vs 'mean' bug), no fallback fitting, leakage-safe fit/transform,
result/report consistency, RFF seed policy, deterministic temporal subsets, timestamp-based bootstrap blocks,
the genuine OvR-vs-OvO ablation, master-pipeline completeness, absence of absolute paths.
"""
import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

import common
from common import (ScalerError, apply_frozen_scaler, calendar_block_ids, load_frozen_scaler, paired_block_bootstrap,
                    select_rff_policy, select_with_tie_rule, temporal_block_subset)

PKG = Path(__file__).resolve().parent.parent


# --- scaler key handling / no fallback fit ------------------------------------------------------------------
def test_scaler_accepts_stage_a_keys(tmp_path):
    np.savez(tmp_path / "s.npz", mean=np.arange(3.), scale=np.ones(3) * 2, var=np.ones(3), n_samples_seen=5)
    m, s = load_frozen_scaler(tmp_path / "s.npz")
    assert np.array_equal(m, np.arange(3.)) and np.array_equal(s, np.full(3, 2.))


def test_scaler_accepts_legacy_sklearn_keys(tmp_path):
    np.savez(tmp_path / "s.npz", mean_=np.zeros(2), scale_=np.ones(2))
    assert load_frozen_scaler(tmp_path / "s.npz")[0].shape == (2,)


def test_scaler_missing_file_raises_never_fits(tmp_path):
    with pytest.raises(ScalerError, match="refusing to fit"):
        load_frozen_scaler(tmp_path / "nope.npz")


def test_scaler_wrong_keys_and_bad_values_raise(tmp_path):
    np.savez(tmp_path / "a.npz", foo=np.zeros(2), bar=np.ones(2))
    with pytest.raises(ScalerError):
        load_frozen_scaler(tmp_path / "a.npz")
    np.savez(tmp_path / "b.npz", mean=np.zeros(2), scale=np.array([1.0, 0.0]))
    with pytest.raises(ScalerError):
        load_frozen_scaler(tmp_path / "b.npz")


def test_saved_scalers_load(out_dir):
    files = sorted((out_dir / "models").glob("scaler_*.npz"))
    if not files:
        pytest.skip("no scalers")
    assert len(files) == 8
    for f in files:
        m, s = load_frozen_scaler(f)
        assert np.isfinite(m).all() and (s > 0).all()


def test_stage_b_source_has_no_fallback_fit_and_uses_strict_loader():
    src = (PKG / "source" / "s04_stage_b_rbf_svc.py").read_text()
    assert "load_frozen_scaler" in src and "StandardScaler" not in src and '"mean_"' not in src
    tree = ast.parse(src)
    fit_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") in ("fit", "fit_transform")]
    # the only .fit is the SVC fit inside the training-subset worker (X, y are the TRAIN subset)
    assert len(fit_calls) == 1


# --- leakage-safe fit/transform -----------------------------------------------------------------------------
def test_transform_uses_frozen_train_statistics_only(tmp_path, monkeypatch):
    from sklearn import preprocessing
    rng = np.random.RandomState(0)
    tr, va = rng.randn(500, 4), rng.randn(200, 4) * 3 + 10          # validation deliberately shifted
    mean, scale = tr.mean(0), tr.std(0)
    np.savez(tmp_path / "s.npz", mean=mean, scale=scale)

    def boom(*a, **k):
        raise AssertionError("a scaler must not be fitted while applying a frozen scaler")
    monkeypatch.setattr(preprocessing.StandardScaler, "fit", boom)
    m, s = load_frozen_scaler(tmp_path / "s.npz")
    Z = apply_frozen_scaler(va.astype(np.float32), m, s)
    assert np.allclose(Z, (va.astype(np.float32).astype(np.float64) - mean) / scale)
    assert abs(Z.mean()) > 5            # a validation-fitted scaler would have centred this at 0
    with pytest.raises(ScalerError):
        apply_frozen_scaler(va[:, :3], m, s)


# --- RFF seed policy ----------------------------------------------------------------------------------------
def _synthetic_stage_c(vals):
    rows = []
    for (cfg, D, seed), v in vals.items():
        rows.append({"config": cfg, "D": D, "seed": seed, "gamma": 0.0005, "C": 10.0, "val_scopeB_macro_f1": v, "val_trans_f1": 0.3})
    return pd.DataFrame(rows)


def test_rff_policy_never_best_or_worst_seed_and_is_deterministic():
    seeds = [42, 43, 44, 45, 46]
    pca = dict(zip(seeds, [0.70, 0.75, 0.72, 0.74, 0.69]))
    hos = dict(zip(seeds, [0.68, 0.73, 0.71, 0.72, 0.70]))
    vals = {("pca_B", 2048, s): pca[s] for s in seeds} | {("hosvd_B", 2048, s): hos[s] for s in seeds}
    vals |= {("pca_B", 256, s): 0.5 for s in seeds} | {("hosvd_B", 256, s): 0.5 for s in seeds}
    df = _synthetic_stage_c(vals)
    r1, r2 = select_rff_policy(df), select_rff_policy(df.sample(frac=1, random_state=3))
    assert r1["D_star"] == 2048 and r1 == r2
    order = r1["seed_rank_best_to_worst"]
    assert r1["seed_star"] == order[2] and r1["seed_star"] not in (order[0], order[-1])
    assert r1["chosen"]["pca_B"]["seed"] == r1["chosen"]["hosvd_B"]["seed"]          # shared frequencies for the pair


def test_rff_policy_picks_D_by_cross_seed_mean_not_best_single_run():
    seeds = [42, 43, 44, 45, 46]
    v = {("pca_B", 1024, s): 0.70 for s in seeds} | {("hosvd_B", 1024, s): 0.70 for s in seeds}
    v |= {("pca_B", 2048, s): (0.95 if s == 42 else 0.60) for s in seeds} | {("hosvd_B", 2048, s): 0.60 for s in seeds}
    assert select_rff_policy(_synthetic_stage_c(v))["D_star"] == 1024          # one lucky 0.95 run must not win


def test_revised_manifest_matches_policy_recomputation(need_results):
    out = need_results("manifests/REVISED_SELECTION_post_audit.json", "results/stage_c_full_experiment.csv")
    rev = json.loads((out / "manifests/REVISED_SELECTION_post_audit.json").read_text())
    pol = select_rff_policy(pd.read_csv(out / "results/stage_c_full_experiment.csv"))
    assert (rev["D_star"], rev["seed_star"]) == (pol["D_star"], pol["seed_star"])
    assert rev["seed_star"] not in (pol["seed_rank_best_to_worst"][0], pol["seed_rank_best_to_worst"][-1])
    assert rev["predeclared_seeds"] == [42, 43, 44, 45, 46] and rev["test_metrics_used_for_selection"] is False
    orig = json.loads((out / "manifests/LOCKED_SELECTION.json").read_text())      # original lock must be untouched
    assert orig["selected_models"]["rff_pca_B"]["seed"] == 43


def test_tie_rule_prefers_transition_f1_within_tolerance():
    rows = [{"val_scopeB_macro_f1": 0.770, "val_trans_f1": 0.30, "val_trans_recall": .4, "val_scopeB_balanced_accuracy": .7, "cost": 1, "id": "a"},
            {"val_scopeB_macro_f1": 0.766, "val_trans_f1": 0.38, "val_trans_recall": .4, "val_scopeB_balanced_accuracy": .7, "cost": 9, "id": "b"},
            {"val_scopeB_macro_f1": 0.700, "val_trans_f1": 0.90, "val_trans_recall": .9, "val_scopeB_balanced_accuracy": .9, "cost": 0, "id": "c"}]
    assert select_with_tie_rule(rows)["id"] == "b"          # c is outside the 0.005 band and must not win


# --- Stage B subsets ----------------------------------------------------------------------------------------
def test_temporal_subset_deterministic_sorted_and_spread():
    a, b = temporal_block_subset(24108, 4000), temporal_block_subset(24108, 4000)
    assert np.array_equal(a, b) and len(a) == 4000 and len(np.unique(a)) == 4000 and np.all(np.diff(a) > 0)
    for q in range(4):                                         # every quarter of the training period represented
        lo, hi = q * 24108 / 4, (q + 1) * 24108 / 4
        assert ((a >= lo) & (a < hi)).sum() > 500
    assert len(temporal_block_subset(24108, 8000)) == 8000
    with pytest.raises(ValueError):
        temporal_block_subset(24108, 4050)


def test_stage_b_outputs_status_consistency(need_results):
    out = need_results("results/stage_b_all_candidates.csv", "results/stage_b_subsets.csv")
    df = pd.read_csv(out / "results/stage_b_all_candidates.csv")
    assert len(df) == 36 and not df.duplicated(["config", "fit_n", "gamma_scale", "C"]).any()
    assert set(df["status"]) <= {"done", "skipped_timeout", "skipped_ram_guard", "skipped_budget", "failed_error"}
    if (df.status != "done").any():                                                   # every non-run cell must carry a reason
        assert "skip_reason" in df.columns and df.loc[df.status != "done", "skip_reason"].notna().all()
    for rid in df.loc[df.status == "done", "run_id"]:
        assert (out / "checkpoints/stage_b" / f"{rid}_valpred.npz").exists() or (out / "checkpoints").exists() is False
    sub = pd.read_csv(out / "results/stage_b_subsets.csv")
    assert set(sub["fit_n"]) == {4000, 8000} and (sub["distinct_calendar_months_covered"] == sub["full_train_calendar_months"]).all()


def test_ovr_ablation_is_genuinely_different_from_dfs_toggle():
    rng = np.random.RandomState(0)
    X = np.vstack([rng.randn(80, 5) + m for m in ([0] * 5, [2] * 5, [-2, 2, 0, 0, 1])])
    y = np.repeat([0, 1, 2], 80)
    ovo = SVC(kernel="rbf", gamma=0.1, C=1, decision_function_shape="ovr").fit(X, y)
    ovo2 = SVC(kernel="rbf", gamma=0.1, C=1, decision_function_shape="ovo").fit(X, y)
    ovr = OneVsRestClassifier(SVC(kernel="rbf", gamma=0.1, C=1)).fit(X, y)
    assert np.array_equal(ovo.predict(X), ovo2.predict(X))          # the old 'ablation' changed nothing
    assert len(ovr.estimators_) == 3 and all(e.n_support_.sum() > 0 for e in ovr.estimators_)
    assert ovo.n_support_.shape == (3,)                              # native OvO: 3 pairwise problems, support counted per class
    assert sum(e.n_support_.sum() for e in ovr.estimators_) != ovo.n_support_.sum()


# --- bootstrap ----------------------------------------------------------------------------------------------
def test_calendar_blocks_follow_real_time_not_row_index():
    ts = np.concatenate([np.datetime64("2016-07-02T00") + np.arange(30) * np.timedelta64(1, "h"),
                         np.datetime64("2016-07-10T00") + np.arange(30) * np.timedelta64(1, "h")])   # 8-day gap
    ids = calendar_block_ids(ts, 24)
    assert ids[0] == ids[23] != ids[24] and ids.max() + 1 == 4                      # 24h blocks; gap does not create empty blocks
    assert calendar_block_ids(ts, 24 * 7).max() + 1 == 2


def test_bootstrap_matches_sklearn_and_is_deterministic():
    rng = np.random.RandomState(1)
    n = 3000
    y = rng.randint(0, 3, n)
    c = np.where(rng.rand(n) < 0.05, -1, rng.randint(0, 3, n))
    pa = np.where(rng.rand(n) < 0.7, y, rng.randint(0, 3, n))
    pb = np.where(rng.rand(n) < 0.6, y, rng.randint(0, 3, n))
    ts = np.datetime64("2016-07-02T00") + np.arange(n) * np.timedelta64(1, "h")
    r1 = paired_block_bootstrap(y, c, {"a": pa, "b": pb}, ts, [("a", "b")], n_boot=300, seed=5)
    r2 = paired_block_bootstrap(y, c, {"a": pa, "b": pb}, ts, [("a", "b")], n_boot=300, seed=5)
    assert r1 == r2
    obs = r1["pairs"]["a - b"]["macro_f1"]["observed_diff_a_minus_b"]
    assert obs == pytest.approx(f1_score(y, pa, average="macro") - f1_score(y, pb, average="macro"), abs=1e-12)
    same = paired_block_bootstrap(y, c, {"a": pa, "b": pa}, ts, [("a", "b")], n_boot=100, seed=1)["pairs"]["a - b"]["trans_f1"]
    assert same["ci95"] == [0.0, 0.0] and same["ci_includes_zero"]


def test_bootstrap_artifact_not_overclaimed(need_results):
    out = need_results("results/bootstrap_ci.json", "reports/phase5_research_report.md")
    B = json.loads((out / "results/bootstrap_ci.json").read_text())["runs"]["block_24h"]["pairs"]
    rep = (out / "reports/phase5_research_report.md").read_text()
    tf = B["linear_pca_B - linear_hosvd_B"]["trans_f1"]
    if tf["ci_includes_zero"]:
        assert "transition-F1 difference is statistically" not in rep.lower()
        assert f"{tf['ci95'][0]:+.4f}" in rep and f"{tf['ci95'][1]:+.4f}" in rep       # actual interval is quoted
    assert "[+0.0068, +0.0664]" not in rep and "[+0.0264, +0.0601]" not in rep         # stale pre-audit intervals must be gone


# --- consistency of generated artifacts with prediction CSVs -------------------------------------------------
def test_metrics_table_matches_prediction_csvs_and_original_eval(need_results):
    out = need_results("results/final_metrics_from_predictions.csv", "results/reconciliation_vs_final_evaluation.csv")
    rec = pd.read_csv(out / "results/reconciliation_vs_final_evaluation.csv")
    assert rec["max_abs_diff"].max() < 1e-9
    tab = pd.read_csv(out / "results/final_metrics_from_predictions.csv")
    for mid in ("linear_pca_B", "linear_best_overall", "revised_rff_pca_B"):
        r = tab[(tab.model_id == mid) & (tab.split == "test")]
        if r.empty:
            continue
        fam = "predictions/revised" if mid.startswith("revised") else "predictions"
        d = pd.read_csv(out / fam / f"final_{mid}_test.csv")
        assert r["scopeB_macro_f1"].iloc[0] == pytest.approx(f1_score(d.y, d.hard_pred, average="macro"), abs=1e-12)
    pca_c = tab[(tab.model_id == "linear_best_overall") & (tab.split == "test")]["scopeB_macro_f1"].iloc[0]
    assert round(pca_c, 4) == 0.7907                       # the figure flagged by the audit


def test_audit_summary_and_report_are_generated_from_tables(need_results):
    out = need_results("audit_summary.json", "results/final_metrics_from_predictions.csv", "reports/phase5_research_report.md")
    A = json.loads((out / "audit_summary.json").read_text())
    tab = pd.read_csv(out / "results/final_metrics_from_predictions.csv")
    rep = (out / "reports/phase5_research_report.md").read_text()
    for mid, v in A["test_results_summary"]["scopeB_macro_f1"].items():
        t = tab[(tab.model_id == mid) & (tab.split == "test")]
        assert not t.empty and v == pytest.approx(float(t["scopeB_macro_f1"].iloc[0]), abs=1e-9), mid
        assert f"{v:.4f}" in rep, f"{mid} test macro-F1 {v:.4f} missing from report"
    assert A["generated_from"] == "results/*.csv|json (no hard-coded metrics)"


def test_report_does_not_carry_known_stale_values(need_results):
    out = need_results("reports/phase5_research_report.md")
    rep = (out / "reports/phase5_research_report.md").read_text()
    if "**Correction.**" in rep:                       # the correction paragraph legitimately quotes the withdrawn claim
        head, tail = rep.split("**Correction.**", 1)
        rep = head + tail.split("\n\n", 1)[1]
    for stale in ("0.7656", "0.7770", "0.7226", "Independently Audited", "statistically robust"):
        assert stale not in rep, stale


# --- pipeline completeness / portability --------------------------------------------------------------------
def test_master_pipeline_all_includes_stage_b_and_is_ordered():
    sys.path.insert(0, str(PKG))
    import phase5_run
    plan = phase5_run.build_plan("all")
    assert "stage_b" in plan and plan.index("stage_a") < plan.index("stage_b") < plan.index("eval")
    assert plan.index("eval") < plan.index("revision") < plan.index("tables") < plan.index("report") < plan.index("test") < plan.index("package")
    assert phase5_run.build_plan("stage_b") == ["stage_b"]


def test_no_absolute_local_paths_anywhere():
    bad = []
    for f in list((PKG / "source").glob("*.py")) + list((PKG / "tests").glob("*.py")) + [PKG / "phase5_run.py"]:
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if re.search(r"/Users/|C:\\\\|/home/[a-z]+/", line) and "re.search" not in line:
                bad.append(f"{f.name}:{i}")
    assert not bad, bad


def test_common_metrics_equal_original_s06_implementations():
    import s06_selection_and_evaluation as s06
    rng = np.random.RandomState(3)
    y, p = rng.randint(0, 3, 4000), rng.randint(0, 3, 4000)
    c = np.where(rng.rand(4000) < 0.1, -1, rng.randint(0, 3, 4000))
    a, b = s06.compute_scope_a_transition_metrics(y, p, c), common.transition_metrics(y, p, c)
    for k in ("true_transitions", "pred_transitions", "trans_tp", "trans_fp", "trans_fn", "trans_tn", "high_tp", "high_fp", "high_fn"):
        assert a[k] == b[k], k
    for k in ("trans_precision", "trans_recall", "trans_f1", "trans_dest_accuracy", "high_precision", "high_recall", "high_f1"):
        assert a[k] == pytest.approx(b[k], abs=1e-12), k
    P = rng.dirichlet(np.ones(3), 4000)
    pa, pb = s06.compute_prob_metrics(y, P), common.prob_metrics(y, P)
    assert pa["brier_score"] == pytest.approx(pb["brier"], abs=1e-12) and pa["log_loss"] == pytest.approx(pb["log_loss"], abs=1e-12)


def test_reported_test_summary_is_not_hardcoded(out_dir):
    """audit_summary's test counts, if present, must equal the pytest JUnit result (no literal 'passed: 14')."""
    ts, au = out_dir / "results/test_summary.json", out_dir / "audit_summary.json"
    if not (ts.exists() and au.exists()):
        pytest.skip("test summary not yet generated")
    T, A = json.loads(ts.read_text()), json.loads(au.read_text())
    if A["tests"].get("source") == "results/test_summary.json":
        assert (A["tests"]["total"], A["tests"]["failed"], A["tests"]["skipped"]) == (T["total"], T["failed"], T["skipped"])
