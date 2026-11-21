"""
plotting.py – Generate Phase 6 scientific figures.
Handles missing models gracefully (log warning, don't fabricate).
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_sns_style():
    plt.style.use("seaborn-v0_8-whitegrid")


def fig_accuracy_macro_f1(out_path: Path):
    """fig01: fair Markov/SVM/fusion accuracy vs macro-F1 comparison."""
    make_sns_style()
    res_dir = out_path.parent.parent / "results"
    df = pd.read_csv(res_dir / "final_evaluation.csv")
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(df))
    width = 0.35
    ax.bar([i - width/2 for i in x], df["accuracy"].values, width, label="Accuracy", color="#2b5c8f")
    ax.bar([i + width/2 for i in x], df["macro_f1"].values, width, label="Macro-F1", color="#f28e2b")
    ax.set_xticks(x)
    labels = [m.replace("linear_", "").replace("stageB_full_", "RBF_") for m in df["model_id"].values]
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title("Scope A Accuracy vs Macro-F1 Comparison")
    ax.set_ylim(0.5, 1.0)
    ax.legend(loc="lower right")
    ax.grid(True, ls=":", alpha=0.5)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close()


def fig_transition_detection(out_path: Path):
    """fig02: binary change precision/recall/F1 alongside A_change and A_stable."""
    make_sns_style()
    res_dir = out_path.parent.parent / "results"
    df = pd.read_csv(res_dir / "transition_detection_metrics.csv")
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(df))
    w = 0.2
    ax.bar([i - 1.5*w for i in x], df["change_precision"].values, w, label="Change Prec", color="#d62728")
    ax.bar([i - 0.5*w for i in x], df["change_recall"].values, w, label="Change Recall", color="#ff9896")
    ax.bar([i + 0.5*w for i in x], df["change_F1"].values, w, label="Change F1", color="#9467bd")
    ax.plot(x, df["A_change"].values, "go--", label="A_change (exact dest)", ms=6, lw=1.5)
    ax.plot(x, df["A_stable"].values, "mo--", label="A_stable", ms=6, lw=1.5)
    ax.set_xticks(x)
    labels = [m.replace("linear_", "").replace("stageB_full_", "RBF_") for m in df["model_id"].values]
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.set_title("Scope A Transition Detection & Exact Accuracy (A_change vs Change Recall)")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close()


def fig_fixed_lambda_sensitivity(out_path: Path):
    """fig03: lambda sensitivity on validation."""
    make_sns_style()
    res_dir = out_path.parent.parent / "results"
    df = pd.read_csv(res_dir / "fixed_fusion_grid.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    colors = {"linear_pca_B": "#2b5c8f", "linear_hosvd_B": "#4ba3e3", "linear_best_overall": "#2ca02c", "revised_rff_pca_B": "#f28e2b"}
    metrics = [("macro_f1", "Macro-F1"), ("change_F1", "Change-F1"), ("accuracy", "Accuracy")]
    sources = df["svm_source"].unique()

    for i, (col, name) in enumerate(metrics):
        ax = axes[i]
        for src in sources:
            sub = df[df["svm_source"] == src]
            ax.plot(sub["lambda"].values, sub[col].values, "o-", color=colors.get(src, "#888888"), label=src, lw=1.5, ms=4)
        ax.set_xlabel("λ (Fixed Weight)")
        ax.set_ylabel(name)
        ax.set_title(f"Validation {name} vs λ")
        ax.grid(True, ls=":", alpha=0.5)
        if i == 0:
            ax.legend(fontsize=7, loc="lower right")

    fig.suptitle("Experiment F1: Validation Sensitivity to Fixed Weight λ", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close()


def fig_adaptive_thresholds(out_path: Path):
    """fig04: tau/weight sensitivity on validation."""
    make_sns_style()
    res_dir = out_path.parent.parent / "results"
    df = pd.read_csv(res_dir / "adaptive_fusion_grid.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    sources = df["svm_source"].unique()
    colors = ["#2b5c8f", "#f28e2b", "#2ca02c", "#d62728", "#9467bd"]

    for idx, src in enumerate(sources):
        sub = df[df["svm_source"] == src]
        tau_grouped = sub.groupby("tau")["J"].max()
        ax.plot(tau_grouped.index.values, tau_grouped.values, "o-", label=f"Max J: {src}", color=colors[idx % len(colors)], lw=1.5)

    ax.set_xlabel("τ (Transition Risk Threshold)")
    ax.set_ylabel("Validation J Score")
    ax.set_title("Experiment F2: Max Validation J Score vs Risk Threshold τ")
    ax.legend(fontsize=8)
    ax.grid(True, ls=":", alpha=0.5)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close()


def fig_confusion_matrices(out_path: Path):
    """fig05: confusion matrices for key models."""
    make_sns_style()
    res_dir = out_path.parent.parent / "results"
    cm_df = pd.read_csv(res_dir / "confusion_matrices.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    model_ids = cm_df["model_id"].values[:3]

    for ax, mid in zip(axes, model_ids):
        row = cm_df[cm_df["model_id"] == mid]
        if not row.empty:
            cm = eval(row.iloc[0]["confusion_matrix"])
            cm_arr = np.array(cm)
            im = ax.imshow(cm_arr, cmap="Blues", vmin=0, vmax=cm_arr.max())
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center", color="white" if cm_arr[i, j] > cm_arr.max()/2 else "black")
            short_title = mid.split("_")[0] + "_" + mid.split("_")[1]
            ax.set_title(short_title, fontsize=10)
            ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
            ax.set_xticklabels(["Low", "Mod", "High"]); ax.set_yticklabels(["Low", "Mod", "High"])
            ax.set_xlabel("Predicted Class"); ax.set_ylabel("True Class")

    plt.suptitle("Test Scope A Three-Class Confusion Matrices", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close()


def fig_transition_risk(out_path: Path):
    """fig06: risk distributions for truly stable vs changing hours."""
    make_sns_style()
    pred_path = out_path.parent.parent / "predictions/test/test_predictions_merged.csv"
    if pred_path.exists():
        df = pd.read_csv(pred_path)
        mask_a = df["scope_a_eligible"].astype(bool)
        actual_change = (df["true_next_state"] != df["observed_current_state"]).astype(bool)
        # Compute SVM transition risk r_S,t = 1 - p_S,t(S_t)
        cs = df["observed_current_state"].values
        p_svm = df[["p_svm_0", "p_svm_1", "p_svm_2"]].values
        r_svm = np.zeros(len(df))
        for i in range(len(df)):
            if cs[i] >= 0:
                r_svm[i] = 1.0 - p_svm[i, int(cs[i])]

        stable_risk = r_svm[mask_a & ~actual_change]
        changed_risk = r_svm[mask_a & actual_change]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(stable_risk, bins=35, alpha=0.6, label=f"Truly Stable (n={len(stable_risk)})", color="#2b5c8f", density=True)
        ax.hist(changed_risk, bins=35, alpha=0.6, label=f"Truly Changing (n={len(changed_risk)})", color="#d62728", density=True)
        ax.set_xlabel("SVM Transition Risk r_S,t = 1 − p_S,t(S_t)")
        ax.set_ylabel("Density")
        ax.set_title("Test Scope A: Distribution of SVM Transition Risk r_S,t")
        ax.legend()
        ax.grid(True, ls=":", alpha=0.5)
        plt.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close()


def fig_accuracy_change_tradeoff(out_path: Path):
    """fig07: accuracy vs change-F1 scatter with Pareto candidates."""
    make_sns_style()
    res_dir = out_path.parent.parent / "results"
    df = pd.read_csv(res_dir / "final_evaluation.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2b5c8f", "#f28e2b", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for idx, row in df.iterrows():
        mid = row["model_id"]
        acc = row["accuracy"]
        ch_f1 = row["change_F1"]
        ax.scatter(acc, ch_f1, s=70, color=colors[idx % len(colors)], label=mid[:25])
        ax.annotate(mid[:18], (acc, ch_f1), xytext=(5, 5), textcoords="offset points", fontsize=8)

    ax.set_xlabel("Overall Accuracy (Scope A)")
    ax.set_ylabel("Binary Change F1 (Scope A)")
    ax.set_title("Scope A Model Comparison: Accuracy vs Change Detection Trade-off")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(fontsize=7, loc="lower left")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close()


def fig_probability_calibration(out_path: Path):
    """fig08: reliability / probability quality comparison."""
    make_sns_style()
    res_dir = out_path.parent.parent / "results"
    df = pd.read_csv(res_dir / "probability_quality.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    valid_df = df.dropna(subset=["brier", "log_loss"])
    x = range(len(valid_df))
    labels = [m[:18] for m in valid_df["model_id"].values]

    ax1.bar(x, valid_df["brier"].values, color="#2b5c8f", width=0.4)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax1.set_ylabel("Brier Score"); ax1.set_title("Multiclass Brier Score (Lower is Better)")

    ax2.bar(x, valid_df["log_loss"].values, color="#f28e2b", width=0.4)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax2.set_ylabel("Log Loss"); ax2.set_title("Multiclass Log Loss (Lower is Better)")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close()


def fig_computational_cost(out_path: Path):
    """fig09: Phase 6 overhead by approach."""
    make_sns_style()
    res_dir = out_path.parent.parent / "results"
    cost_df = pd.read_csv(res_dir / "computational_cost.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(cost_df))
    bars = ax.bar(x, cost_df["inference_time_s"].values, width=0.4, color="#2b5c8f")
    ax.set_xticks(x)
    ax.set_xticklabels(cost_df["model_id"].values, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Execution Time (seconds)")
    ax.set_title("Phase 6 Computational Cost & Runtime Overhead")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f"{yval:.2f}s", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close()


FIGURES = {
    "fig01_accuracy_macro_f1.png": fig_accuracy_macro_f1,
    "fig02_transition_detection.png": fig_transition_detection,
    "fig03_fixed_lambda_sensitivity.png": fig_fixed_lambda_sensitivity,
    "fig04_adaptive_thresholds.png": fig_adaptive_thresholds,
    "fig05_confusion_matrices.png": fig_confusion_matrices,
    "fig06_transition_risk.png": fig_transition_risk,
    "fig07_accuracy_change_tradeoff.png": fig_accuracy_change_tradeoff,
    "fig08_probability_calibration.png": fig_probability_calibration,
    "fig09_computational_cost.png": fig_computational_cost,
}


def generate_all(figures_dir):
    figures_dir = Path(figures_dir)
    for fname, fn in FIGURES.items():
        out_path = figures_dir / fname
        try:
            fn(out_path)
            print(f"Generated figure: {fname}")
        except Exception as e:
            print(f"WARN: figure {fname} failed: {e}")
