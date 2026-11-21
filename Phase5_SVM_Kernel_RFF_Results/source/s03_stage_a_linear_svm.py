#!/usr/bin/env python3
"""
s03_stage_a_linear_svm.py — Stage A: Full-Train Linear SVM (40 Fits)
Phase 5 SVM/RFF Air Pollution Classification

Tasks:
  1. Section 4: Fit StandardScaler on FULL training F only for all 8 representations.
     Save frozen scalers to models/scaler_{config}.npz.
  2. Section 6: Fit 40 LinearSVC candidates (8 configs x 5 C values: 0.01, 0.1, 1, 10, 100).
     Evaluate on validation Scope A and Scope B.
     Compute classification, transition, and high-entry metrics.
  3. Save results, C-curves, confusion matrices, and checkpoints.
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_recall_fscore_support
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

CONFIGS = [
    "pca_A", "pca_B", "pca_C",
    "hosvd_A", "hosvd_B", "hosvd_C",
    "hosvd_B_alt1", "hosvd_B_alt2"
]
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 5 Stage A: Linear SVM 40 fits")
    parser.add_argument("--input-dir", type=str, default=".", help="Path to input files directory")
    parser.add_argument("--output-dir", type=str, default="output", help="Path to output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-iter", type=int, default=5000, help="Max iterations for LinearSVC")
    return parser.parse_args()


def compute_transition_metrics(y_true, y_pred, current_state):
    """
    Scope A transition metrics where current_state >= 0.
    T_i = 1[y_{i,t+1} != c_{i,t}]
    T_hat_i = 1[yhat_{i,t+1} != c_{i,t}]
    """
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

    # Destination accuracy among actual transitions
    actual_trans_mask = (T == 1)
    if np.sum(actual_trans_mask) > 0:
        dest_acc = float(np.mean(y_true_s[actual_trans_mask] == y_pred_s[actual_trans_mask]))
    else:
        dest_acc = 0.0

    # Accuracy on stable rows
    stable_mask = (T == 0)
    if np.sum(stable_mask) > 0:
        stable_acc = float(np.mean(y_true_s[stable_mask] == y_pred_s[stable_mask]))
    else:
        stable_acc = 0.0

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
        "trans_tp": tp, "trans_fp": fp, "trans_fn": fn, "trans_tn": tn,
        "trans_precision": prec, "trans_recall": rec, "trans_f1": f1,
        "trans_dest_accuracy": dest_acc, "trans_stable_accuracy": stable_acc,
        "high_tp": h_tp, "high_fp": h_fp, "high_fn": h_fn,
        "high_precision": h_prec, "high_recall": h_rec, "high_f1": h_f1
    }


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    log_dir = output_dir / "logs"
    models_dir = output_dir / "models"
    results_dir = output_dir / "results"
    figures_dir = output_dir / "figures"
    checkpoints_dir = output_dir / "checkpoints"

    for d in [log_dir, models_dir, results_dir, figures_dir, checkpoints_dir]:
        d.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "s03_stage_a.log"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger("stage_a")
    logger.info("Starting Phase 5 Stage A Linear SVM fits...")
    logger.info(f"Input dir: {input_dir}, Output dir: {output_dir}, Seed: {args.seed}")

    # Step 1: Preprocessing & Scaler fitting
    logger.info("=" * 60)
    logger.info("STEP 1: Fitting StandardScalers on FULL training data only")
    logger.info("=" * 60)

    scalers = {}
    scaled_data = {}

    for cfg in CONFIGS:
        train_path = input_dir / f"features_train_{cfg}.npz"
        val_path = input_dir / f"features_validation_{cfg}.npz"
        test_path = input_dir / f"features_test_{cfg}.npz"

        with np.load(train_path) as d:
            F_train = d["F"].astype(np.float32)
            y_train = d["y"].astype(np.int8)
            cs_train = d["current_state"].astype(np.int8)

        with np.load(val_path) as d:
            F_val = d["F"].astype(np.float32)
            y_val = d["y"].astype(np.int8)
            cs_val = d["current_state"].astype(np.int8)

        # Fit on TRAIN ONLY
        scaler = StandardScaler()
        scaler.fit(F_train)
        # Safe handling of constant columns: where scale_ == 0 or var_ == 0, set scale to 1.0
        zero_std_mask = (scaler.scale_ == 0.0) | np.isnan(scaler.scale_)
        if np.any(zero_std_mask):
            logger.warning(f"Config {cfg} has {np.sum(zero_std_mask)} constant feature columns. Setting scale to 1.0.")
            scaler.scale_[zero_std_mask] = 1.0

        # Save scaler parameters
        scaler_file = models_dir / f"scaler_{cfg}.npz"
        np.savez_compressed(
            scaler_file,
            mean=scaler.mean_,
            scale=scaler.scale_,
            var=scaler.var_,
            n_samples_seen=scaler.n_samples_seen_
        )

        F_train_scaled = scaler.transform(F_train).astype(np.float32)
        F_val_scaled = scaler.transform(F_val).astype(np.float32)

        assert np.all(np.isfinite(F_train_scaled)), f"Non-finite in scaled train {cfg}"
        assert np.all(np.isfinite(F_val_scaled)), f"Non-finite in scaled val {cfg}"

        scalers[cfg] = scaler
        scaled_data[cfg] = {
            "F_train": F_train_scaled, "y_train": y_train, "cs_train": cs_train,
            "F_val": F_val_scaled, "y_val": y_val, "cs_val": cs_val
        }
        logger.info(f"Fitted and saved scaler for {cfg}: dim={F_train.shape[1]}, train_N={F_train.shape[0]}")

    # Step 2: 40 LinearSVC fits
    logger.info("=" * 60)
    logger.info(f"STEP 2: Running 40 LinearSVC Fits (8 configs x {len(C_GRID)} C values)")
    logger.info("=" * 60)

    results_rows = []
    best_per_config = {}

    for cfg in CONFIGS:
        logger.info(f"--- Fitting models for representation: {cfg} ---")
        F_train = scaled_data[cfg]["F_train"]
        y_train = scaled_data[cfg]["y_train"]
        F_val = scaled_data[cfg]["F_val"]
        y_val = scaled_data[cfg]["y_val"]
        cs_val = scaled_data[cfg]["cs_val"]

        scope_a_val_mask = (cs_val >= 0)

        best_score = -1.0
        best_row = None

        for c_val in C_GRID:
            start_time = time.time()
            rss_before = psutil.Process().memory_info().rss / (1024 * 1024)

            clf = LinearSVC(
                C=c_val,
                loss="squared_hinge",
                multi_class="ovr",
                max_iter=args.max_iter,
                random_state=args.seed,
                dual=False,
                tol=1e-4
            )
            clf.fit(F_train, y_train)

            fit_time = time.time() - start_time
            rss_after = psutil.Process().memory_info().rss / (1024 * 1024)
            n_iter = int(np.max(clf.n_iter_)) if hasattr(clf, "n_iter_") else -1

            # Predictions on Validation
            pred_val = clf.predict(F_val)

            # Scope B Metrics (Full validation split: 4239)
            acc_b = accuracy_score(y_val, pred_val)
            macro_f1_b = f1_score(y_val, pred_val, average="macro", zero_division=0)
            weighted_f1_b = f1_score(y_val, pred_val, average="weighted", zero_division=0)
            bal_acc_b = balanced_accuracy_score(y_val, pred_val)

            p_b, r_b, f_b, _ = precision_recall_fscore_support(y_val, pred_val, labels=[0, 1, 2], zero_division=0)

            # Scope A Metrics (current_state >= 0: 4197)
            y_val_a = y_val[scope_a_val_mask]
            pred_val_a = pred_val[scope_a_val_mask]
            acc_a = accuracy_score(y_val_a, pred_val_a)
            macro_f1_a = f1_score(y_val_a, pred_val_a, average="macro", zero_division=0)
            weighted_f1_a = f1_score(y_val_a, pred_val_a, average="weighted", zero_division=0)
            bal_acc_a = balanced_accuracy_score(y_val_a, pred_val_a)

            # Transition & High entry metrics on Scope A
            trans_metrics = compute_transition_metrics(y_val, pred_val, cs_val)

            row = {
                "config": cfg,
                "C": c_val,
                "fit_time_sec": fit_time,
                "n_iter": n_iter,
                "rss_mb": rss_after,
                # Scope B
                "val_scopeB_accuracy": acc_b,
                "val_scopeB_macro_f1": macro_f1_b,
                "val_scopeB_weighted_f1": weighted_f1_b,
                "val_scopeB_balanced_accuracy": bal_acc_b,
                "val_scopeB_f1_low": f_b[0],
                "val_scopeB_f1_mod": f_b[1],
                "val_scopeB_f1_high": f_b[2],
                # Scope A
                "val_scopeA_accuracy": acc_a,
                "val_scopeA_macro_f1": macro_f1_a,
                "val_scopeA_weighted_f1": weighted_f1_a,
                "val_scopeA_balanced_accuracy": bal_acc_a,
                # Transitions
                "val_trans_tp": trans_metrics["trans_tp"],
                "val_trans_fp": trans_metrics["trans_fp"],
                "val_trans_fn": trans_metrics["trans_fn"],
                "val_trans_tn": trans_metrics["trans_tn"],
                "val_trans_precision": trans_metrics["trans_precision"],
                "val_trans_recall": trans_metrics["trans_recall"],
                "val_trans_f1": trans_metrics["trans_f1"],
                "val_trans_dest_accuracy": trans_metrics["trans_dest_accuracy"],
                "val_trans_stable_accuracy": trans_metrics["trans_stable_accuracy"],
                # High Entry
                "val_high_tp": trans_metrics["high_tp"],
                "val_high_fp": trans_metrics["high_fp"],
                "val_high_fn": trans_metrics["high_fn"],
                "val_high_precision": trans_metrics["high_precision"],
                "val_high_recall": trans_metrics["high_recall"],
                "val_high_f1": trans_metrics["high_f1"],
            }
            results_rows.append(row)

            # Save checkpoint
            ckpt_path = checkpoints_dir / f"stage_a_{cfg}_C{c_val}.pkl"
            with open(ckpt_path, "wb") as f:
                pickle.dump(clf, f)

            logger.info(
                f"[{cfg} | C={c_val}] ScopeB Macro-F1={macro_f1_b:.4f}, ScopeB Acc={acc_b:.4f}, "
                f"ScopeA Trans-F1={trans_metrics['trans_f1']:.4f}, Time={fit_time:.2f}s, Iter={n_iter}"
            )

            # Check if best candidate for this config
            # Primary: validation Scope B macro-F1
            # Tie breaking within 0.005: Scope A transition F1, transition recall, balanced accuracy, compute cost
            if best_row is None:
                best_row = row
                best_score = macro_f1_b
            else:
                score_diff = macro_f1_b - best_score
                if score_diff > 0.005:
                    best_row = row
                    best_score = macro_f1_b
                elif abs(score_diff) <= 0.005:
                    # Tie break: trans F1
                    if row["val_trans_f1"] > best_row["val_trans_f1"]:
                        best_row = row
                        best_score = macro_f1_b
                    elif row["val_trans_f1"] == best_row["val_trans_f1"]:
                        if row["val_trans_recall"] > best_row["val_trans_recall"]:
                            best_row = row
                            best_score = macro_f1_b
                        elif row["val_trans_recall"] == best_row["val_trans_recall"]:
                            if row["val_scopeB_balanced_accuracy"] > best_row["val_scopeB_balanced_accuracy"]:
                                best_row = row
                                best_score = macro_f1_b
                            elif row["fit_time_sec"] < best_row["fit_time_sec"]:
                                best_row = row
                                best_score = macro_f1_b

        best_per_config[cfg] = best_row
        logger.info(f"Winner for {cfg}: C={best_row['C']} with Scope B Macro-F1 = {best_row['val_scopeB_macro_f1']:.4f}")

    # Save complete 40-row candidate results
    df_all = pd.DataFrame(results_rows)
    all_csv_path = results_dir / "stage_a_all_candidates.csv"
    df_all.to_csv(all_csv_path, index=False)
    logger.info(f"Saved 40 candidate results to: {all_csv_path}")

    # Save 8 best per-config results
    df_best = pd.DataFrame(list(best_per_config.values()))
    best_csv_path = results_dir / "stage_a_best_per_config.csv"
    df_best.to_csv(best_csv_path, index=False)
    logger.info(f"Saved 8 best-per-config results to: {best_csv_path}")

    # Generate Figures
    # 1. Validation C curves
    plt.figure(figsize=(10, 6))
    for cfg in CONFIGS:
        cfg_rows = [r for r in results_rows if r["config"] == cfg]
        c_vals = [r["C"] for r in cfg_rows]
        f1_vals = [r["val_scopeB_macro_f1"] for r in cfg_rows]
        linestyle = "--" if "hosvd" in cfg else "-"
        marker = "s" if "hosvd" in cfg else "o"
        plt.plot(c_vals, f1_vals, label=cfg, linestyle=linestyle, marker=marker, linewidth=1.5)

    plt.xscale("log")
    plt.xlabel("Regularization parameter C")
    plt.ylabel("Validation Scope B Macro-F1")
    plt.title("Stage A Linear SVM: Validation Scope B Macro-F1 vs C")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    c_curves_fig = figures_dir / "stage_a_c_curves.png"
    plt.savefig(c_curves_fig, dpi=300)
    plt.close()
    logger.info(f"Saved C-curves plot to: {c_curves_fig}")

    # 2. Confusion Matrices for the 8 best validation models
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, cfg in enumerate(CONFIGS):
        winner = best_per_config[cfg]
        best_c = winner["C"]
        ckpt_path = checkpoints_dir / f"stage_a_{cfg}_C{best_c}.pkl"
        with open(ckpt_path, "rb") as f:
            clf = pickle.load(f)

        pred_val = clf.predict(scaled_data[cfg]["F_val"])
        cm = confusion_matrix(scaled_data[cfg]["y_val"], pred_val, labels=[0, 1, 2])

        ax = axes[idx]
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        ax.set_title(f"{cfg} (C={best_c})\nMacro-F1={winner['val_scopeB_macro_f1']:.4f}")
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(["Low", "Mod", "High"])
        ax.set_yticklabels(["Low", "Mod", "High"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.tight_layout()
    cm_fig = figures_dir / "stage_a_confusion_matrices.png"
    plt.savefig(cm_fig, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrices to: {cm_fig}")

    logger.info("Stage A completed successfully!")


if __name__ == "__main__":
    main()

