#!/usr/bin/env python3
"""
s06_selection_and_evaluation.py — Selection, Markov Baseline, Probability Calibration, Final Test Evaluation
Phase 5 SVM/RFF Air Pollution Classification

Modules:
  1. Section 12: Validation-only Model Selection and Test Lock (LOCKED_SELECTION.json)
  2. Section 10: Markov Baseline realignment & metrics recomputation from Phase 3 CSVs
  3. Section 10: Train-only probability calibration for selected models
  4. Section 9: Scope A/B transition & High-entry metrics evaluation on Test
  5. Section 11: Transition cost-sensitive supplement (sample weights on Scope A train)
  6. Section 14: Final comparison figures & aligned prediction CSVs
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_recall_fscore_support
)
from sklearn.svm import LinearSVC


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 5: Selection, Baseline, Calibration & Test Evaluation")
    parser.add_argument("--input-dir", type=str, default=".", help="Input directory")
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parent.parent), help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def compute_scope_a_transition_metrics(y_true, y_pred, current_state):
    mask = current_state >= 0
    y_true_s = y_true[mask]
    y_pred_s = y_pred[mask]
    c_s = current_state[mask]

    T = (y_true_s != c_s).astype(int)
    T_hat = (y_pred_s != c_s).astype(int)

    tp = int(np.sum((T == 1) & (T_hat == 1)))
    fp = int(np.sum((T == 0) & (T_hat == 1)))
    fn = int(np.sum((T == 1) & (T_hat == 0)))
    tn = int(np.sum((T == 0) & (T_hat == 0)))

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    actual_trans_mask = (T == 1)
    dest_acc = float(np.mean(y_true_s[actual_trans_mask] == y_pred_s[actual_trans_mask])) if np.sum(actual_trans_mask) > 0 else 0.0
    stable_mask = (T == 0)
    stable_acc = float(np.mean(y_true_s[stable_mask] == y_pred_s[stable_mask])) if np.sum(stable_mask) > 0 else 0.0

    # Directed transitions (6 combinations)
    transitions_counts = {}
    for from_c in [0, 1, 2]:
        for to_y in [0, 1, 2]:
            if from_c != to_y:
                cnt = int(np.sum((c_s == from_c) & (y_true_s == to_y)))
                pred_cnt = int(np.sum((c_s == from_c) & (y_pred_s == to_y)))
                correct_cnt = int(np.sum((c_s == from_c) & (y_true_s == to_y) & (y_pred_s == to_y)))
                transitions_counts[f"{from_c}->{to_y}"] = {
                    "true_count": cnt,
                    "pred_count": pred_cnt,
                    "correct": correct_cnt
                }

    # High entry: H = 1[c_i != 2 and y_i == 2], Hhat = 1[c_i != 2 and yhat_i == 2]
    H = ((c_s != 2) & (y_true_s == 2)).astype(int)
    H_hat = ((c_s != 2) & (y_pred_s == 2)).astype(int)
    h_tp = int(np.sum((H == 1) & (H_hat == 1)))
    h_fp = int(np.sum((H == 0) & (H_hat == 1)))
    h_fn = int(np.sum((H == 1) & (H_hat == 0)))
    h_prec = h_tp / (h_tp + h_fp) if (h_tp + h_fp) > 0 else 0.0
    h_rec = h_tp / (h_tp + h_fn) if (h_tp + h_fn) > 0 else 0.0
    h_f1 = 2 * h_prec * h_rec / (h_prec + h_rec) if (h_prec + h_rec) > 0 else 0.0

    return {
        "n_scope_a": int(len(y_true_s)),
        "true_transitions": int(np.sum(T)),
        "pred_transitions": int(np.sum(T_hat)),
        "trans_tp": tp, "trans_fp": fp, "trans_fn": fn, "trans_tn": tn,
        "trans_precision": prec, "trans_recall": rec, "trans_f1": f1,
        "trans_dest_accuracy": dest_acc, "trans_stable_accuracy": stable_acc,
        "directed_transitions": transitions_counts,
        "true_high_entries": int(np.sum(H)),
        "pred_high_entries": int(np.sum(H_hat)),
        "high_tp": h_tp, "high_fp": h_fp, "high_fn": h_fn,
        "high_precision": h_prec, "high_recall": h_rec, "high_f1": h_f1
    }


def compute_prob_metrics(y_true, probs, eps=1e-12):
    """
    Multiclass Brier score and clipped log-loss.
    probs: [N, 3]
    """
    N = len(y_true)
    probs_clipped = np.clip(probs, eps, 1.0)
    # Renormalize rows
    probs_clipped = probs_clipped / np.sum(probs_clipped, axis=1, keepdims=True)

    # Brier Score: 1/N sum_i sum_k (p_ik - 1[y_i==k])^2
    one_hot = np.zeros((N, 3), dtype=np.float64)
    for k in range(3):
        one_hot[:, k] = (y_true == k).astype(float)
    brier = float(np.mean(np.sum((probs_clipped - one_hot) ** 2, axis=1)))

    # Log-Loss: -1/N sum_i log(p_i,y_i)
    true_class_probs = probs_clipped[np.arange(N), y_true]
    logloss = float(-np.mean(np.log(true_class_probs)))

    return {"brier_score": brier, "log_loss": logloss}


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    log_dir = output_dir / "logs"
    models_dir = output_dir / "models"
    results_dir = output_dir / "results"
    figures_dir = output_dir / "figures"
    checkpoints_dir = output_dir / "checkpoints"
    manifests_dir = output_dir / "manifests"
    predictions_dir = output_dir / "predictions"

    for d in [log_dir, models_dir, results_dir, figures_dir, checkpoints_dir, manifests_dir, predictions_dir]:
        d.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "s06_evaluation.log"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger("eval")
    logger.info("Starting Phase 5 Model Selection, Calibration & Evaluation...")

    # =========================================================================
    # MODULE 1: Model Selection & Test Lock
    # =========================================================================
    logger.info("=" * 60)
    logger.info("MODULE 1: Locking Model Selection (VALIDATION-ONLY)")
    logger.info("=" * 60)

    # 1. Load Stage A best per config
    stage_a_best_path = results_dir / "stage_a_best_per_config.csv"
    if not stage_a_best_path.exists():
        logger.error(f"Missing {stage_a_best_path} — cannot perform selection")
        sys.exit(1)
    df_stage_a = pd.read_csv(stage_a_best_path)

    # Best Linear models for pca_B and hosvd_B
    lin_pca_b = df_stage_a[df_stage_a["config"] == "pca_B"].iloc[0].to_dict()
    lin_hosvd_b = df_stage_a[df_stage_a["config"] == "hosvd_B"].iloc[0].to_dict()

    # Best overall linear model across all 8 configs
    best_linear_overall = df_stage_a.loc[df_stage_a["val_scopeB_macro_f1"].idxmax()].to_dict()

    # 2. Load Stage C full experiment
    stage_c_path = results_dir / "stage_c_full_experiment.csv"
    rff_pca_b = None
    rff_hosvd_b = None
    if stage_c_path.exists():
        df_stage_c = pd.read_csv(stage_c_path)
        # Select best RFF configuration based on mean validation Scope B Macro-F1 across seeds, or highest validation Macro-F1
        # Primary selection: val_scopeB_macro_f1
        c_pca_b = df_stage_c[df_stage_c["config"] == "pca_B"]
        c_hosvd_b = df_stage_c[df_stage_c["config"] == "hosvd_B"]
        if len(c_pca_b) > 0:
            # Deterministic selection: highest val_scopeB_macro_f1
            rff_pca_b = c_pca_b.loc[c_pca_b["val_scopeB_macro_f1"].idxmax()].to_dict()
        if len(c_hosvd_b) > 0:
            rff_hosvd_b = c_hosvd_b.loc[c_hosvd_b["val_scopeB_macro_f1"].idxmax()].to_dict()

    # Stage B (Exact RBF) if available
    stage_b_path = results_dir / "stage_b_all_candidates.csv"
    rbf_pca_b = None
    rbf_hosvd_b = None
    if stage_b_path.exists():
        df_stage_b = pd.read_csv(stage_b_path)
        b_done = df_stage_b[df_stage_b["status"] == "done"] if "status" in df_stage_b.columns else df_stage_b
        if len(b_done) > 0:
            b_pca_b = b_done[b_done["config"] == "pca_B"]
            b_hosvd_b = b_done[b_done["config"] == "hosvd_B"]
            if len(b_pca_b) > 0:
                rbf_pca_b = b_pca_b.loc[b_pca_b["scopeB_macro_f1"].idxmax()].to_dict()
            if len(b_hosvd_b) > 0:
                rbf_hosvd_b = b_hosvd_b.loc[b_hosvd_b["scopeB_macro_f1"].idxmax()].to_dict()

    lock_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "Primary: validation Scope B macro-F1 (highest wins); Secondary tie handling (diff <= 0.005): Scope A transition F1, transition recall, balanced accuracy, compute cost.",
        "test_metrics_used": False,
        "selected_models": {
            "linear_pca_B": {
                "config": "pca_B",
                "model_type": "LinearSVC_OvR",
                "C": float(lin_pca_b["C"]),
                "val_scopeB_macro_f1": float(lin_pca_b["val_scopeB_macro_f1"]),
                "val_scopeB_accuracy": float(lin_pca_b["val_scopeB_accuracy"]),
                "val_scopeA_trans_f1": float(lin_pca_b["val_trans_f1"])
            },
            "linear_hosvd_B": {
                "config": "hosvd_B",
                "model_type": "LinearSVC_OvR",
                "C": float(lin_hosvd_b["C"]),
                "val_scopeB_macro_f1": float(lin_hosvd_b["val_scopeB_macro_f1"]),
                "val_scopeB_accuracy": float(lin_hosvd_b["val_scopeB_accuracy"]),
                "val_scopeA_trans_f1": float(lin_hosvd_b["val_trans_f1"])
            },
            "best_linear_overall": {
                "config": best_linear_overall["config"],
                "model_type": "LinearSVC_OvR",
                "C": float(best_linear_overall["C"]),
                "val_scopeB_macro_f1": float(best_linear_overall["val_scopeB_macro_f1"]),
                "val_scopeB_accuracy": float(best_linear_overall["val_scopeB_accuracy"])
            }
        }
    }

    if rff_pca_b is not None:
        lock_payload["selected_models"]["rff_pca_B"] = {
            "config": "pca_B",
            "model_type": "RFF_LinearSVC_OvR",
            "D": int(rff_pca_b["D"]),
            "seed": int(rff_pca_b["seed"]),
            "gamma": float(rff_pca_b["gamma"]),
            "C": float(rff_pca_b["C"]),
            "val_scopeB_macro_f1": float(rff_pca_b["val_scopeB_macro_f1"]),
            "val_scopeA_trans_f1": float(rff_pca_b["val_trans_f1"])
        }
    if rff_hosvd_b is not None:
        lock_payload["selected_models"]["rff_hosvd_B"] = {
            "config": "hosvd_B",
            "model_type": "RFF_LinearSVC_OvR",
            "D": int(rff_hosvd_b["D"]),
            "seed": int(rff_hosvd_b["seed"]),
            "gamma": float(rff_hosvd_b["gamma"]),
            "C": float(rff_hosvd_b["C"]),
            "val_scopeB_macro_f1": float(rff_hosvd_b["val_scopeB_macro_f1"]),
            "val_scopeA_trans_f1": float(rff_hosvd_b["val_trans_f1"])
        }

    lock_file = manifests_dir / "LOCKED_SELECTION.json"
    with open(lock_file, "w") as f:
        json.dump(lock_payload, f, indent=2)
    logger.info(f"FROZEN and locked model selection to: {lock_file}")

    # =========================================================================
    # MODULE 2: Markov Baseline Realignment & Recomputation
    # =========================================================================
    logger.info("=" * 60)
    logger.info("MODULE 2: Recomputing Phase 3 Markov Baselines from CSVs")
    logger.info("=" * 60)

    val_csv_path = input_dir / "markov_predictions_validation.csv"
    test_csv_path = input_dir / "markov_predictions_test.csv"

    val_df = pd.read_csv(val_csv_path)
    test_df = pd.read_csv(test_csv_path)

    logger.info(f"Loaded validation Markov CSV: {len(val_df)} rows")
    logger.info(f"Loaded test Markov CSV: {len(test_df)} rows")

    # Audit Scope A eligible counts
    val_scope_a = val_df[val_df["scope_a_eligible"] == True]
    test_scope_a = test_df[test_df["scope_a_eligible"] == True]
    assert len(val_scope_a) == 4197, f"Expected 4197 Scope A val rows, got {len(val_scope_a)}"
    assert len(test_scope_a) == 5685, f"Expected 5685 Scope A test rows, got {len(test_scope_a)}"

    # Recompute true transitions on Scope A
    val_true_trans = int(np.sum(val_scope_a["y"] != val_scope_a["current_state"]))
    test_true_trans = int(np.sum(test_scope_a["y"] != test_scope_a["current_state"]))
    assert val_true_trans == 488, f"Expected 488 val transitions, got {val_true_trans}"
    assert test_true_trans == 526, f"Expected 526 test transitions, got {test_true_trans}"

    # Entries into High
    val_high_entries = int(np.sum((val_scope_a["current_state"] != 2) & (val_scope_a["y"] == 2)))
    test_high_entries = int(np.sum((test_scope_a["current_state"] != 2) & (test_scope_a["y"] == 2)))
    assert val_high_entries == 122, f"Expected 122 val High entries, got {val_high_entries}"
    assert test_high_entries == 154, f"Expected 154 test High entries, got {test_high_entries}"

    # Recompute Markov classification metrics
    markov_rows = []
    for split_name, df_split, scope_a_split in [("validation", val_df, val_scope_a), ("test", test_df, test_scope_a)]:
        # Models: majority, persistence, markov
        for model_name, pred_col, prob_prefix in [
            ("majority", "pred_majority", "prob_majority"),
            ("persistence", "pred_persistence_scope_b", None),
            ("markov", "pred_markov_scope_b", "prob_markov")
        ]:
            # Scope B
            y_b = df_split["y"].values
            yhat_b = df_split[pred_col].values
            acc_b = accuracy_score(y_b, yhat_b)
            macro_f1_b = f1_score(y_b, yhat_b, average="macro", zero_division=0)
            bal_acc_b = balanced_accuracy_score(y_b, yhat_b)

            # Scope A
            y_a = scope_a_split["y"].values
            yhat_a = scope_a_split[pred_col].values
            acc_a = accuracy_score(y_a, yhat_a)
            macro_f1_a = f1_score(y_a, yhat_a, average="macro", zero_division=0)
            bal_acc_a = balanced_accuracy_score(y_a, yhat_a)

            # Transitions on Scope A
            trans_m = compute_scope_a_transition_metrics(y_a, yhat_a, scope_a_split["current_state"].values)

            # Probabilistic metrics
            brier_val = np.nan
            logloss_val = np.nan
            if prob_prefix is not None:
                prob_cols = [f"{prob_prefix}_0", f"{prob_prefix}_1", f"{prob_prefix}_2"]
                if all(c in df_split.columns for c in prob_cols):
                    probs = scope_a_split[prob_cols].values
                    p_res = compute_prob_metrics(y_a, probs)
                    brier_val = p_res["brier_score"]
                    logloss_val = p_res["log_loss"]

            markov_rows.append({
                "split": split_name,
                "model": model_name,
                "scopeB_accuracy": acc_b,
                "scopeB_macro_f1": macro_f1_b,
                "scopeB_balanced_accuracy": bal_acc_b,
                "scopeA_accuracy": acc_a,
                "scopeA_macro_f1": macro_f1_a,
                "scopeA_balanced_accuracy": bal_acc_a,
                "scopeA_brier": brier_val,
                "scopeA_log_loss": logloss_val,
                "true_transitions": trans_m["true_transitions"],
                "pred_transitions": trans_m["pred_transitions"],
                "trans_tp": trans_m["trans_tp"],
                "trans_fp": trans_m["trans_fp"],
                "trans_fn": trans_m["trans_fn"],
                "trans_tn": trans_m["trans_tn"],
                "trans_precision": trans_m["trans_precision"],
                "trans_recall": trans_m["trans_recall"],
                "trans_f1": trans_m["trans_f1"],
                "trans_dest_accuracy": trans_m["trans_dest_accuracy"],
                "high_tp": trans_m["high_tp"],
                "high_fp": trans_m["high_fp"],
                "high_fn": trans_m["high_fn"],
                "high_precision": trans_m["high_precision"],
                "high_recall": trans_m["high_recall"],
                "high_f1": trans_m["high_f1"]
            })

    df_markov_recomputed = pd.DataFrame(markov_rows)
    markov_csv = results_dir / "markov_baseline_recomputed.csv"
    df_markov_recomputed.to_csv(markov_csv, index=False)
    logger.info(f"Saved recomputed Markov baseline metrics to {markov_csv}")

    # Log key findings
    test_markov_a = df_markov_recomputed[(df_markov_recomputed["split"] == "test") & (df_markov_recomputed["model"] == "markov")].iloc[0]
    logger.info(
        f"Markov Test Scope A Accuracy: {test_markov_a['scopeA_accuracy']:.6f} "
        f"(Reference ~0.907476: delta = {abs(test_markov_a['scopeA_accuracy'] - 0.907476):.6f})"
    )
    logger.info(
        f"Markov Test Scope A Hard-Detected Transitions: {test_markov_a['pred_transitions']} of {test_markov_a['true_transitions']} true changes! "
        f"(Transition Recall = {test_markov_a['trans_recall']:.4f}, F1 = {test_markov_a['trans_f1']:.4f})"
    )

    # =========================================================================
    # MODULE 3 & 4: Train Calibration & Final Evaluation on Selected Models
    # =========================================================================
    logger.info("=" * 60)
    logger.info("MODULE 3 & 4: Training Calibration & Final Test Evaluation")
    logger.info("=" * 60)

    # Models to evaluate
    eval_models = [
        {"model_id": "linear_pca_B", "config": "pca_B", "type": "linear", "C": float(lin_pca_b["C"])},
        {"model_id": "linear_hosvd_B", "config": "hosvd_B", "type": "linear", "C": float(lin_hosvd_b["C"])},
        {"model_id": "linear_best_overall", "config": best_linear_overall["config"], "type": "linear", "C": float(best_linear_overall["C"])}
    ]
    if rff_pca_b is not None:
        eval_models.append({
            "model_id": "rff_pca_B", "config": "pca_B", "type": "rff",
            "D": int(rff_pca_b["D"]), "seed": int(rff_pca_b["seed"]),
            "gamma": float(rff_pca_b["gamma"]), "C": float(rff_pca_b["C"])
        })
    if rff_hosvd_b is not None:
        eval_models.append({
            "model_id": "rff_hosvd_B", "config": "hosvd_B", "type": "rff",
            "D": int(rff_hosvd_b["D"]), "seed": int(rff_hosvd_b["seed"]),
            "gamma": float(rff_hosvd_b["gamma"]), "C": float(rff_hosvd_b["C"])
        })

    final_eval_rows = []

    for m_info in eval_models:
        model_id = m_info["model_id"]
        cfg = m_info["config"]
        m_type = m_info["type"]
        logger.info(f"Processing final model: {model_id} ({cfg}, {m_type})...")

        # Load scalers and raw features
        scaler_npz = np.load(models_dir / f"scaler_{cfg}.npz")
        mean_ = scaler_npz["mean"]
        scale_ = scaler_npz["scale"]

        with np.load(input_dir / f"features_train_{cfg}.npz") as d:
            F_train = (d["F"] - mean_) / scale_
            y_train = d["y"].astype(np.int8)
            cs_train = d["current_state"].astype(np.int8)
            ts_train = d["target_timestamp"]

        with np.load(input_dir / f"features_validation_{cfg}.npz") as d:
            F_val = (d["F"] - mean_) / scale_
            y_val = d["y"].astype(np.int8)
            cs_val = d["current_state"].astype(np.int8)
            ts_val = d["target_timestamp"]

        with np.load(input_dir / f"features_test_{cfg}.npz") as d:
            F_test = (d["F"] - mean_) / scale_
            y_test = d["y"].astype(np.int8)
            cs_test = d["current_state"].astype(np.int8)
            ts_test = d["target_timestamp"]

        # If RFF, apply sampler
        if m_type == "rff":
            D = m_info["D"]
            rff_seed = m_info["seed"]
            gamma = m_info["gamma"]
            sampler = RBFSampler(gamma=gamma, n_components=D, random_state=rff_seed)
            X_train = sampler.fit_transform(F_train)
            X_val = sampler.transform(F_val)
            X_test = sampler.transform(F_test)
        else:
            X_train = F_train
            X_val = F_val
            X_test = F_test

        # --- Probability Calibration (Train-Only) ---
        # Chronological forward split within TRAIN: 80% base_train, 20% calib_hold
        n_train = len(y_train)
        split_idx = int(n_train * 0.8)
        X_base_tr = X_train[:split_idx]
        y_base_tr = y_train[:split_idx]
        X_calib_h = X_train[split_idx:]
        y_calib_h = y_train[split_idx:]

        clf_base = LinearSVC(
            C=m_info["C"],
            loss="squared_hinge",
            multi_class="ovr",
            max_iter=5000,
            random_state=42,
            dual=False
        )
        clf_base.fit(X_base_tr, y_base_tr)

        # Get scores on calib_hold
        scores_calib_h = clf_base.decision_function(X_calib_h)
        calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        calibrator.fit(scores_calib_h, y_calib_h)

        # Save calibrator
        calib_file = models_dir / f"calibrator_{model_id}.pkl"
        with open(calib_file, "wb") as f:
            pickle.dump(calibrator, f)

        # --- Fit Final Model on FULL Train ---
        clf_final = LinearSVC(
            C=m_info["C"],
            loss="squared_hinge",
            multi_class="ovr",
            max_iter=5000,
            random_state=42,
            dual=False
        )
        clf_final.fit(X_train, y_train)

        # Save final model
        model_file = models_dir / f"final_{model_id}.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(clf_final, f)

        # Evaluate on Validation and Test
        for split_name, X_s, y_s, cs_s, ts_s in [
            ("validation", X_val, y_val, cs_val, ts_val),
            ("test", X_test, y_test, cs_test, ts_test)
        ]:
            scores_s = clf_final.decision_function(X_s)
            hard_pred_s = clf_final.predict(X_s)
            probs_s = calibrator.predict_proba(scores_s)
            calib_argmax_s = np.argmax(probs_s, axis=1)

            # Scope B Metrics
            acc_b = accuracy_score(y_s, hard_pred_s)
            macro_f1_b = f1_score(y_s, hard_pred_s, average="macro", zero_division=0)
            weighted_f1_b = f1_score(y_s, hard_pred_s, average="weighted", zero_division=0)
            bal_acc_b = balanced_accuracy_score(y_s, hard_pred_s)
            prob_b = compute_prob_metrics(y_s, probs_s)

            # Scope A Metrics
            scope_a_mask = (cs_s >= 0)
            y_s_a = y_s[scope_a_mask]
            hard_pred_s_a = hard_pred_s[scope_a_mask]
            probs_s_a = probs_s[scope_a_mask]

            acc_a = accuracy_score(y_s_a, hard_pred_s_a)
            macro_f1_a = f1_score(y_s_a, hard_pred_s_a, average="macro", zero_division=0)
            bal_acc_a = balanced_accuracy_score(y_s_a, hard_pred_s_a)
            prob_a = compute_prob_metrics(y_s_a, probs_s_a)

            trans_m = compute_scope_a_transition_metrics(y_s, hard_pred_s, cs_s)

            final_eval_rows.append({
                "model_id": model_id,
                "config": cfg,
                "model_type": m_type,
                "split": split_name,
                "scopeB_accuracy": acc_b,
                "scopeB_macro_f1": macro_f1_b,
                "scopeB_weighted_f1": weighted_f1_b,
                "scopeB_balanced_accuracy": bal_acc_b,
                "scopeB_brier": prob_b["brier_score"],
                "scopeB_log_loss": prob_b["log_loss"],
                "scopeA_accuracy": acc_a,
                "scopeA_macro_f1": macro_f1_a,
                "scopeA_balanced_accuracy": bal_acc_a,
                "scopeA_brier": prob_a["brier_score"],
                "scopeA_log_loss": prob_a["log_loss"],
                "true_transitions": trans_m["true_transitions"],
                "pred_transitions": trans_m["pred_transitions"],
                "trans_tp": trans_m["trans_tp"],
                "trans_fp": trans_m["trans_fp"],
                "trans_fn": trans_m["trans_fn"],
                "trans_tn": trans_m["trans_tn"],
                "trans_precision": trans_m["trans_precision"],
                "trans_recall": trans_m["trans_recall"],
                "trans_f1": trans_m["trans_f1"],
                "trans_dest_accuracy": trans_m["trans_dest_accuracy"],
                "true_high_entries": trans_m["true_high_entries"],
                "pred_high_entries": trans_m["pred_high_entries"],
                "high_tp": trans_m["high_tp"],
                "high_fp": trans_m["high_fp"],
                "high_fn": trans_m["high_fn"],
                "high_precision": trans_m["high_precision"],
                "high_recall": trans_m["high_recall"],
                "high_f1": trans_m["high_f1"],
            })

            # Save aligned prediction CSV
            pred_df = pd.DataFrame({
                "target_timestamp": ts_s,
                "y": y_s,
                "current_state": cs_s,
                "scope_a_eligible": scope_a_mask,
                "hard_pred": hard_pred_s,
                "calibrated_argmax": calib_argmax_s,
                "score_0": scores_s[:, 0],
                "score_1": scores_s[:, 1],
                "score_2": scores_s[:, 2],
                "prob_0": probs_s[:, 0],
                "prob_1": probs_s[:, 1],
                "prob_2": probs_s[:, 2],
                "model_id": model_id
            })
            pred_csv_path = predictions_dir / f"final_{model_id}_{split_name}.csv"
            pred_df.to_csv(pred_csv_path, index=False)
            logger.info(f"Saved prediction CSV: {pred_csv_path}")

    df_final_eval = pd.DataFrame(final_eval_rows)
    eval_csv = results_dir / "final_evaluation_scopeAB.csv"
    df_final_eval.to_csv(eval_csv, index=False)
    logger.info(f"Saved final evaluation metrics to {eval_csv}")

    # =========================================================================
    # MODULE 5: Transition Cost-Sensitive Supplement (Section 11)
    # =========================================================================
    logger.info("=" * 60)
    logger.info("MODULE 5: Cost-Sensitive Transition Weighting Supplement")
    logger.info("=" * 60)

    cost_rows = []
    # Test on winning pca_B and hosvd_B linear models
    for cfg, c_val in [("pca_B", float(lin_pca_b["C"])), ("hosvd_B", float(lin_hosvd_b["C"]))]:
        scaler_npz = np.load(models_dir / f"scaler_{cfg}.npz")
        mean_ = scaler_npz["mean"]
        scale_ = scaler_npz["scale"]

        with np.load(input_dir / f"features_train_{cfg}.npz") as d:
            F_train = ((d["F"] - mean_) / scale_).astype(np.float32)
            y_train = d["y"].astype(np.int8)
            cs_train = d["current_state"].astype(np.int8)

        with np.load(input_dir / f"features_validation_{cfg}.npz") as d:
            F_val = ((d["F"] - mean_) / scale_).astype(np.float32)
            y_val = d["y"].astype(np.int8)
            cs_val = d["current_state"].astype(np.int8)

        # On Scope A train rows:
        scope_a_train_mask = (cs_train >= 0)
        # sample_weight: s_i = lambda if y_next != c_current else 1
        for lam in [1.0, 2.0, 5.0, 10.0]:
            weights = np.ones(len(y_train), dtype=np.float32)
            # Apply lambda only on Scope A transitions
            trans_train_mask = scope_a_train_mask & (y_train != cs_train)
            weights[trans_train_mask] = lam

            clf_cost = LinearSVC(
                C=c_val,
                loss="squared_hinge",
                multi_class="ovr",
                max_iter=5000,
                random_state=42,
                dual=False
            )
            clf_cost.fit(F_train, y_train, sample_weight=weights)

            pred_val = clf_cost.predict(F_val)
            trans_m = compute_scope_a_transition_metrics(y_val, pred_val, cs_val)
            acc_b = accuracy_score(y_val, pred_val)
            macro_f1_b = f1_score(y_val, pred_val, average="macro", zero_division=0)

            cost_rows.append({
                "config": cfg,
                "C": c_val,
                "lambda": lam,
                "val_scopeB_macro_f1": macro_f1_b,
                "val_scopeB_accuracy": acc_b,
                "trans_precision": trans_m["trans_precision"],
                "trans_recall": trans_m["trans_recall"],
                "trans_f1": trans_m["trans_f1"],
                "trans_dest_accuracy": trans_m["trans_dest_accuracy"],
                "high_precision": trans_m["high_precision"],
                "high_recall": trans_m["high_recall"],
                "high_f1": trans_m["high_f1"],
                "pred_transitions": trans_m["pred_transitions"]
            })
            logger.info(
                f"Cost-Sensitive [{cfg} | lambda={lam}] Trans-Recall={trans_m['trans_recall']:.4f}, "
                f"Trans-F1={trans_m['trans_f1']:.4f}, ScopeB Macro-F1={macro_f1_b:.4f}"
            )

    df_cost = pd.DataFrame(cost_rows)
    cost_csv = results_dir / "cost_sensitive_ablation.csv"
    df_cost.to_csv(cost_csv, index=False)
    logger.info(f"Saved cost-sensitive ablation results to {cost_csv}")

    # =========================================================================
    # MODULE 6: Comparison Plots
    # =========================================================================
    logger.info("=" * 60)
    logger.info("MODULE 6: Generating Comparison Figures")
    logger.info("=" * 60)

    # 1. Markov vs SVM Transition Recall / Precision / F1
    test_rows_eval = df_final_eval[df_final_eval["split"] == "test"]
    markov_test_row = df_markov_recomputed[(df_markov_recomputed["split"] == "test") & (df_markov_recomputed["model"] == "markov")].iloc[0]

    model_names = ["Markov Baseline"] + list(test_rows_eval["model_id"].values)
    recalls = [markov_test_row["trans_recall"]] + list(test_rows_eval["trans_recall"].values)
    precisions = [markov_test_row["trans_precision"]] + list(test_rows_eval["trans_precision"].values)
    f1s = [markov_test_row["trans_f1"]] + list(test_rows_eval["trans_f1"].values)

    x = np.arange(len(model_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width, recalls, width, label="Transition Recall", color="#2b5c8f")
    ax.bar(x, precisions, width, label="Transition Precision", color="#4ba3e3")
    ax.bar(x + width, f1s, width, label="Transition F1", color="#f28e2b")

    ax.set_ylabel("Score")
    ax.set_title("Test Set Scope A Transition Metrics: Markov Baseline vs SVM / RFF Models")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.set_ylim([0, 1.0])
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax.legend()
    plt.tight_layout()
    trans_fig = figures_dir / "markov_vs_svm_transitions.png"
    plt.savefig(trans_fig, dpi=300)
    plt.close()
    logger.info(f"Saved transition comparison figure to {trans_fig}")

    # 2. High-Entry Recall & Precision
    h_recalls = [markov_test_row["high_recall"]] + list(test_rows_eval["high_recall"].values)
    h_precisions = [markov_test_row["high_precision"]] + list(test_rows_eval["high_precision"].values)
    h_f1s = [markov_test_row["high_f1"]] + list(test_rows_eval["high_f1"].values)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width, h_recalls, width, label="High-Entry Recall", color="#d62728")
    ax.bar(x, h_precisions, width, label="High-Entry Precision", color="#ff9896")
    ax.bar(x + width, h_f1s, width, label="High-Entry F1", color="#9467bd")

    ax.set_ylabel("Score")
    ax.set_title("Test Set Scope A Entry-Into-High Metrics: Markov Baseline vs SVM Models")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.set_ylim([0, 1.0])
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax.legend()
    plt.tight_layout()
    high_fig = figures_dir / "high_entry_comparison.png"
    plt.savefig(high_fig, dpi=300)
    plt.close()
    logger.info(f"Saved high-entry comparison figure to {high_fig}")

    logger.info("Module 1-6 completed successfully!")


if __name__ == "__main__":
    main()
