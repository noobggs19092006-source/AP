"""
integrity_checks.py – Verify input provenance, shapes, timestamps and scopes.
Fails loudly on mismatches. No silent row drops.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


EXPECTED_SPLIT_COUNTS = {"train": 24108, "validation": 4239, "test": 5720}
EXPECTED_SCOPE_A      = {"train": 23976, "validation": 4197, "test": 5685}
EXPECTED_CLASS_DIST = {
    "train":      {0: 7872, 1: 5832,  2: 10404},
    "validation": {0: 1848, 1: 1114,  2: 1277},
    "test":       {0: 2142, 1: 1202,  2: 2376},
}
REQUIRED_MARKOV_COLS = {
    "target_timestamp", "y", "current_state", "scope_a_eligible",
    "pred_majority", "pred_persistence_scope_b", "pred_markov_scope_b",
    "prob_majority_0", "prob_majority_1", "prob_majority_2",
    "prob_markov_0", "prob_markov_1", "prob_markov_2",
}
REQUIRED_FEATURE_KEYS = {"F", "y", "target_timestamp", "current_state", "current_pm25"}


def fail(msg):
    print(f"FAIL  {msg}", flush=True)
    raise AssertionError(msg)


def warn(msg):
    print(f"WARN  {msg}")


def ok(msg):
    print(f"OK    {msg}")


def compute_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk_size)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_file_exists(path: Path, label: str):
    if not path.exists():
        fail(f"{label} missing: {path}")
    ok(f"{label} exists: {path.name}")


def to_datetime64_ns(arr):
    if arr.dtype == np.dtype("datetime64[ns]"):
        return arr
    return arr.astype("datetime64[ns]")


def load_meta_npz(path: Path, required_keys=None):
    if required_keys is None:
        required_keys = REQUIRED_FEATURE_KEYS
    try:
        with np.load(path, allow_pickle=True) as data:
            out = {}
            for k in required_keys:
                if k not in data:
                    fail(f"{path}: key '{k}' missing; available: {list(data.files)}")
                out[k] = data[k]
            return out
    except Exception as e:
        fail(f"Failed to load {path}: {e}")


# ---------------------------------------------------------------------------
# Core checks
# ---------------------------------------------------------------------------

def check_feature_npz(path: Path, split: str, expected_n: int, expected_k: int):
    meta = load_meta_npz(path)
    F, y, ts, cs, pm = meta["F"], meta["y"], meta["target_timestamp"], meta["current_state"], meta["current_pm25"]

    n, k = F.shape[0], F.shape[1]
    if n != expected_n:
        fail(f"{path}: n_samples={n} != expected {expected_n}")
    if k != expected_k:
        fail(f"{path}: n_features={k} != expected {expected_k}")
    if not np.all(np.isfinite(F)):
        fail(f"{path}: non-finite values in F")
    unique_y = set(y.tolist())
    if not unique_y.issubset({0, 1, 2}):
        fail(f"{path}: unexpected labels {unique_y}")
    for label, exp_count in EXPECTED_CLASS_DIST[split].items():
        actual = int(np.sum(y == label))
        if actual != exp_count:
            fail(f"{path}: class {label} count={actual} != expected {exp_count}")
    mask_neg1 = cs == -1
    mask_nan = np.isnan(pm)
    only_neg1 = int(np.sum(mask_neg1 & ~mask_nan))
    only_nan  = int(np.sum(mask_nan  & ~mask_neg1))
    if only_neg1 or only_nan:
        fail(f"{path}: state/pm25 mismatch: {only_neg1} neg1-only, {only_nan} nan-only")
    scope_a_n = int(np.sum(cs >= 0))
    if scope_a_n != EXPECTED_SCOPE_A[split]:
        fail(f"{path}: scope_a count={scope_a_n} != expected {EXPECTED_SCOPE_A[split]}")
    ok(f"{path.name}: split={split}, n={n}, k={k}, scope_a={scope_a_n}")
    return to_datetime64_ns(ts), y, cs


def check_markov_csv(path: Path, split: str):
    df = pd.read_csv(path)
    missing = REQUIRED_MARKOV_COLS - set(df.columns)
    if missing:
        fail(f"{path.name}: missing columns {missing}")
    if len(df) != EXPECTED_SPLIT_COUNTS[split]:
        fail(f"{path.name}: rows={len(df)} != expected {EXPECTED_SPLIT_COUNTS[split]}")
    # prob_markov must sum to 1 (within tolerance)
    prob_cols = ["prob_markov_0", "prob_markov_1", "prob_markov_2"]
    probs = df[prob_cols].values
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
        fail(f"{path.name}: prob_markov does not sum to 1")
    ok(f"{path.name}: {len(df)} rows, prob_markov simplex OK")
    return df


def align_timestamps(ts_a, ts_b, label_a, label_b):
    if not np.array_equal(ts_a, ts_b):
        diff_mask = ts_a != ts_b
        n_diff = int(diff_mask.sum())
        fail(f"Timestamp misalignment between {label_a} and {label_b}: {n_diff} differences")
    ok(f"Timestamps aligned between {label_a} and {label_b}")


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def run(project_root: Path, output_dir: Path):
    log_lines = []
    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    log("=" * 60)
    log("PHASE 6 INTEGRITY CHECKS")
    log("=" * 60)

    # --- Phase 3 ---
    P_path = project_root / "Air_Pollution_Phase3_Results/phase3_output/markov_transition_matrix.csv"
    check_file_exists(P_path, "Markov transition matrix")
    df_P = pd.read_csv(P_path)
    # CSV has label header: current_state,next_0_low,next_1_moderate,next_2_high
    P = df_P[[c for c in df_P.columns if c.startswith("next_")]].values.astype(np.float64)
    assert P.shape == (3, 3), f"Markov P shape {P.shape}"
    row_sums = P.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-10):
        fail(f"Markov P rows do not sum to 1: {row_sums}")
    ok("Markov P loaded, shape=(3,3), rows sum to 1")
    P_sha = compute_sha256(P_path)
    log(f"  P sha256={P_sha[:16]}…")

    markov_val = check_markov_csv(project_root / "Air_Pollution_Phase3_Results/phase3_output/markov_predictions_validation.csv", "validation")
    markov_test = check_markov_csv(project_root / "Air_Pollution_Phase3_Results/phase3_output/markov_predictions_test.csv", "test")

    # --- Phase 4 features ---
    feature_paths = [
        ("train",   "pca_B", 192),
        ("validation","pca_B", 192),
        ("test",    "pca_B", 192),
        ("train",   "hosvd_B", 192),
        ("validation","hosvd_B", 192),
        ("test",    "hosvd_B", 192),
        ("train",   "pca_C", 576),
        ("validation","pca_C", 576),
        ("test",    "pca_C", 576),
    ]
    for split, cfg, expected_k in feature_paths:
        fp = project_root / f"phase4_output/features_{split}_{cfg}.npz"
        check_file_exists(fp, f"Feature NPZ {split}/{cfg}")
        check_feature_npz(fp, split, EXPECTED_SPLIT_COUNTS[split], expected_k)

    # --- Timestamp alignment (use pca_B as reference) ---
    ref_ts = {}
    for split in ("train", "validation", "test"):
        fp = project_root / f"phase4_output/features_{split}_pca_B.npz"
        meta = load_meta_npz(fp, ("target_timestamp", "y", "current_state"))
        ref_ts[split] = to_datetime64_ns(meta["target_timestamp"])

    for split, name in [("validation", "markov_val"), ("test", "markov_test")]:
        df_m = markov_val if split == "validation" else markov_test
        ts_m = np.array([np.datetime64(x.replace(" ", "T"), "ns") for x in df_m["target_timestamp"]])
        align_timestamps(ts_m, ref_ts[split], name, f"features_{split}_pca_B")

    # --- Scalers + models existence ---
    scaler_cfgs = ["scaler_pca_B", "scaler_hosvd_B", "scaler_pca_C"]
    model_ids = ["linear_pca_B", "linear_hosvd_B", "linear_best_overall",
                 "revised_rff_pca_B", "revised_rff_hosvd_B",
                 "rff_pca_B", "rff_hosvd_B",
                 "stageB_full_pca_B", "stageB_full_hosvd_B"]
    calibrator_ids = ["calibrator_linear_pca_B", "calibrator_linear_hosvd_B",
                      "calibrator_linear_best_overall",
                      "calibrator_revised_rff_pca_B", "calibrator_revised_rff_hosvd_B",
                      "calibrator_rff_pca_B", "calibrator_rff_hosvd_B"]
    for s in scaler_cfgs:
        check_file_exists(project_root / f"Phase5_SVM_Kernel_RFF_Results/models/{s}.npz", f"Scaler {s}")
    for mid in model_ids:
        mp = project_root / f"Phase5_SVM_Kernel_RFF_Results/models/final_{mid}.pkl"
        check_file_exists(mp, f"Model {mid}")
    for cid in calibrator_ids:
        cp = project_root / f"Phase5_SVM_Kernel_RFF_Results/models/{cid}.pkl"
        check_file_exists(cp, f"Calibrator {cid}")
    # revised mapping files
    for cfg in ["pca_B", "hosvd_B"]:
        check_file_exists(
            project_root / f"Phase5_SVM_Kernel_RFF_Results/models/rff_mapping_revised_{cfg}_D2048_s44.npz",
            f"RFF mapping {cfg}")

    # Save provenance
    prov = {
        "markov_P_sha256": P_sha,
        "timestamp_alignment": "verified",
        "phase6_root": str(output_dir),
    }
    out_prov = output_dir / "input_provenance.json"
    with open(out_prov, "w") as f:
        json.dump(prov, f, indent=2)
    log(f"\nProvenance saved: {out_prov}")

    log("\n" + "=" * 60)
    log("ALL INTEGRITY CHECKS PASSED")
    log("=" * 60)
    return log_lines


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".").resolve()
    out  = Path(sys.argv[2]) if len(sys.argv) > 2 else root / "results"
    lines = run(root, out)
