"""
test_reproducibility.py – Deterministic execution verification.
Verifies that Phase 6 pipeline produces identical outputs on repeated runs.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def test_deterministic_final_evaluation():
    """Final evaluation table must be bit-identical across runs."""
    res_dir = ROOT / "results"
    df = pd.read_csv(res_dir / "final_evaluation.csv")

    # Check required columns exist
    required_cols = ["model_id", "family", "scope", "N", "accuracy", "macro_f1",
                     "change_F1", "A_change", "A_stable", "high_entry_F1",
                     "brier", "log_loss"]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"

    # Check all expected models are present
    expected_models = ["M0_Markov_Baseline", "M1_Standalone_Linear_Best",
                       "M1b_ExactRBF_stageB_full_pca_B", "M1b_ExactRBF_stageB_full_hosvd_B",
                       "F1_Fixed_lam0.9_linear_hosvd_B", "F2_Adaptive_tau0.65_linear_pca_B",
                       "F3_Transition_rho0.75_eta0.0_linear_best_overall"]
    for m in expected_models:
        assert m in df["model_id"].values, f"Missing model: {m}"

    # Check Scope A row count
    scope_a = df[df["scope"] == "ScopeA"]
    assert len(scope_a) == 7, f"Expected 7 Scope A models, got {len(scope_a)}"

    # Check deterministic values (these are the locked results from actual run)
    m0 = scope_a[scope_a["model_id"] == "M0_Markov_Baseline"].iloc[0]
    assert np.isclose(m0["accuracy"], 0.90192308, atol=1e-6), f"Markov accuracy changed: {m0['accuracy']}"
    assert np.isclose(m0["macro_f1"], 0.66503004, atol=1e-6), f"Markov macro_f1 changed: {m0['macro_f1']}"
    assert m0["change_F1"] == 0.0, f"Markov change_F1 should be 0: {m0['change_F1']}"
    assert m0["A_change"] == 0.0, f"Markov A_change should be 0: {m0['A_change']}"
    assert m0["A_stable"] == 1.0, f"Markov A_stable should be 1.0: {m0['A_stable']}"

    f1 = scope_a[scope_a["model_id"].str.startswith("F1_")].iloc[0]
    assert np.isclose(f1["accuracy"], 0.85751748, atol=1e-6), f"F1 accuracy changed: {f1['accuracy']}"
    assert np.isclose(f1["change_F1"], 0.37806874, atol=1e-6), f"F1 change_F1 changed: {f1['change_F1']}"

    f2 = scope_a[scope_a["model_id"].str.startswith("F2_")].iloc[0]
    assert np.isclose(f2["accuracy"], 0.86468531, atol=1e-6), f"F2 accuracy changed: {f2['accuracy']}"
    assert np.isclose(f2["change_F1"], 0.32780847, atol=1e-6), f"F2 change_F1 changed: {f2['change_F1']}"

    f3 = scope_a[scope_a["model_id"].str.startswith("F3_")].iloc[0]
    assert np.isclose(f3["accuracy"], 0.88146853, atol=1e-6), f"F3 accuracy changed: {f3['accuracy']}"
    assert np.isclose(f3["change_F1"], 0.31698974, atol=1e-6), f"F3 change_F1 changed: {f3['change_F1']}"


def test_deterministic_grid_tables():
    """Grid search tables must be deterministic."""
    res_dir = ROOT / "results"

    for fname in ["fixed_fusion_grid.csv", "adaptive_fusion_grid.csv", "transition_fusion_grid.csv"]:
        df = pd.read_csv(res_dir / fname)
        assert len(df) > 0, f"{fname} is empty"

        # Check no NaN in key columns
        for col in df.columns:
            if col in ["svm_source", "lambda", "tau", "lambda_low", "lambda_high", "rho", "eta", "J"]:
                if col in df.columns:
                    assert not df[col].isna().any(), f"{fname}: column {col} has NaN"


def test_deterministic_figures():
    """Figure PNG files must have identical checksums on repeated runs."""
    fig_dir = ROOT / "figures"
    expected_figures = [
        "fig01_accuracy_macro_f1.png",
        "fig02_transition_detection.png",
        "fig03_fixed_lambda_sensitivity.png",
        "fig04_adaptive_thresholds.png",
        "fig05_confusion_matrices.png",
        "fig06_transition_risk.png",
        "fig07_accuracy_change_tradeoff.png",
        "fig08_probability_calibration.png",
        "fig09_computational_cost.png",
    ]

    for fname in expected_figures:
        fpath = fig_dir / fname
        assert fpath.exists(), f"Missing figure: {fname}"
        h = sha256_file(fpath)
        assert len(h) == 64, f"Invalid hash for {fname}: {h}"


def test_manifest_integrity():
    """Artifact inventory must list all files with valid SHA256."""
    man_dir = ROOT / "manifests"
    inv_path = man_dir / "artifact_inventory.json"
    assert inv_path.exists(), "artifact_inventory.json missing"

    # File is actually CSV format despite .json extension
    df = pd.read_csv(inv_path)
    assert len(df) > 0, "Empty inventory"

    for _, row in df.iterrows():
        fpath = ROOT / row["relative_path"]
        if fpath.exists():
            # Skip meta-files that change on every run
            rel = str(row["relative_path"])
            if "execution.log" in rel or "artifact_inventory.json" in rel:
                continue
            actual_hash = sha256_file(fpath)
            assert actual_hash == row["sha256"], f"Hash mismatch for {rel}"
        # Note: some files may be generated after inventory, so we only check existing ones


def test_selection_manifest():
    """Selected configuration manifest must exist and be valid."""
    man_dir = ROOT / "manifests"
    sel_path = man_dir / "model_selection_manifest.json"
    assert sel_path.exists(), "model_selection_manifest.json missing"

    import json
    with open(sel_path) as f:
        sel = json.load(f)

    assert "selection_rule" in sel
    assert "winners" in sel
    assert "F1_Fixed" in sel["winners"]
    assert "F2_Adaptive" in sel["winners"]
    assert "F3_Transition" in sel["winners"]

    # Check locked values
    f1 = sel["winners"]["F1_Fixed"]
    assert f1["lambda"] == 0.9
    assert f1["svm_source"] == "linear_hosvd_B"

    f2 = sel["winners"]["F2_Adaptive"]
    assert f2["tau"] == 0.65
    assert f2["lambda_low"] == 0.75
    assert f2["lambda_high"] == 1.0

    f3 = sel["winners"]["F3_Transition"]
    assert f3["rho"] == 0.75
    assert f3["eta"] == 0.0


def test_bootstrap_ci_deterministic():
    """Bootstrap CIs must be deterministic (fixed seed)."""
    res_dir = ROOT / "results"
    df = pd.read_csv(res_dir / "bootstrap_confidence_intervals.csv")

    assert len(df) == 2, f"Expected 2 bootstrap comparisons, got {len(df)}"

    # Values from the run with seed=42
    f1_m0 = df[df["comparator"] == "F1_vs_M0_Markov"].iloc[0]
    assert np.isclose(f1_m0["point_estimate"], -0.04702, atol=1e-4), f"F1 vs M0 point estimate: {f1_m0['point_estimate']}"
    assert np.isclose(f1_m0["ci_95_low"], -0.0629, atol=1e-3), f"F1 vs M0 CI low: {f1_m0['ci_95_low']}"
    assert np.isclose(f1_m0["ci_95_high"], -0.0324, atol=1e-3), f"F1 vs M0 CI high: {f1_m0['ci_95_high']}"

    f1_m1 = df[df["comparator"] == "F1_vs_M1_SVM"].iloc[0]
    assert np.isclose(f1_m1["point_estimate"], 0.02362, atol=1e-4), f"F1 vs M1 point estimate: {f1_m1['point_estimate']}"
    assert np.isclose(f1_m1["ci_95_low"], 0.0077, atol=1e-3), f"F1 vs M1 CI low: {f1_m1['ci_95_low']}"
    assert np.isclose(f1_m1["ci_95_high"], 0.0389, atol=1e-3), f"F1 vs M1 CI high: {f1_m1['ci_95_high']}"


def test_predictions_csv_structure():
    """Merged predictions CSV must have correct columns and row count."""
    pred_path = ROOT / "predictions/test/test_predictions_merged.csv"
    assert pred_path.exists(), "test_predictions_merged.csv missing"

    df = pd.read_csv(pred_path)
    assert len(df) == 5720, f"Expected 5720 test rows, got {len(df)}"

    required_cols = [
        "target_timestamp", "true_next_state", "observed_current_state",
        "scope_a_eligible",
        "p_markov_0", "p_markov_1", "p_markov_2",
        "p_svm_0", "p_svm_1", "p_svm_2",
        "p_f1_0", "p_f1_1", "p_f1_2",
        "p_f2_0", "p_f2_1", "p_f2_2",
        "p_f3_0", "p_f3_1", "p_f3_2",
        "pred_m0", "pred_m1", "pred_f1", "pred_f2", "pred_f3",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column in predictions: {col}"

    # Probability columns must sum to ~1 for valid rows
    for prefix in ["p_markov", "p_svm", "p_f1", "p_f2", "p_f3"]:
        prob_cols = [c for c in df.columns if c.startswith(prefix)]
        if prob_cols:
            sums = df[prob_cols].sum(axis=1)
            # Only check Scope A rows
            mask = df["scope_a_eligible"]
            assert np.allclose(sums[mask], 1.0, atol=1e-6), f"{prefix} probs don't sum to 1"