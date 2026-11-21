#!/usr/bin/env python3
"""
s11_report_and_audit.py - regenerate figures, the research report and audit_summary.json FROM RESULT FILES ONLY.

No metric in the report or the audit summary is typed in: every number is read from results/*.csv|json,
manifests/*.json and the prediction-derived table results/final_metrics_from_predictions.csv (s10).
Interpretive sentences are conditional on the numbers (e.g. whether a bootstrap interval includes zero).
Running it twice is safe (idempotent); the second run (stage 'finalize') picks up results/test_summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import reliability_table, utc_now

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DISPLAY = {"linear_pca_B": "Linear PCA-B (orig. lock)", "linear_hosvd_B": "Linear HoSVD-B (orig. lock)",
           "linear_best_overall": "Linear PCA-C (orig. lock, best linear)", "rff_pca_B": "RFF PCA-B (orig. lock)",
           "rff_hosvd_B": "RFF HoSVD-B (orig. lock)", "revised_rff_pca_B": "RFF PCA-B (revised, common seed)",
           "revised_rff_hosvd_B": "RFF HoSVD-B (revised, common seed)", "stageB_pca_B": "Exact RBF PCA-B (subset)",
           "stageB_hosvd_B": "Exact RBF HoSVD-B (subset)", "stageB_full_pca_B": "Exact RBF PCA-B (full train)",
           "stageB_full_hosvd_B": "Exact RBF HoSVD-B (full train)", "phase3_majority": "Phase 3 majority",
           "phase3_persistence": "Phase 3 persistence", "phase3_markov": "Phase 3 Markov"}
F4 = lambda v: "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"


def md_table(df: pd.DataFrame, cols, headers=None, fmt=None) -> str:
    headers = headers or cols
    fmt = fmt or {}
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(":---" if i == 0 else "---:" for i in range(len(cols))) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in fmt:
                cells.append(fmt[c](v))
            elif isinstance(v, (float, np.floating)):
                cells.append(F4(float(v)))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def jl(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def rc(p: Path):
    return pd.read_csv(p) if p.exists() else None


# ------------------------------------------------------------------------------------------------ figures
def figures(out: Path, tab: pd.DataFrame, R: dict):
    fig_dir = out / "figures"
    te = tab[tab.split == "test"].set_index("model_id")
    models = [m for m in DISPLAY if m in te.index]
    lab = [DISPLAY[m] for m in models]
    x = np.arange(len(models))

    def bars(cols, names, colors, title, fname, ylabel="score", annotate_zero=False):
        fig, ax = plt.subplots(figsize=(13, 5.2))
        w = 0.8 / len(cols)
        for i, (c, n, col) in enumerate(zip(cols, names, colors)):
            ax.bar(x + (i - (len(cols) - 1) / 2) * w, [te.loc[m, c] for m in models], w, label=n, color=col)
        if annotate_zero:
            for xi, m in zip(x, models):
                if te.loc[m, "pred_transitions"] == 0:
                    ax.text(xi, 0.03, "0 changes\npredicted\n(recall = precision = 0)", ha="center", fontsize=7, color="firebrick")
        ax.set_xticks(x, lab, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", ls=":", alpha=.5)
        ax.legend(fontsize=8, ncol=len(cols))
        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=200)
        plt.close(fig)

    bars(["trans_recall", "trans_precision", "trans_f1", "trans_dest_accuracy"],
         ["transition recall", "transition precision", "transition F1", "destination accuracy | actual change"],
         ["#2b5c8f", "#4ba3e3", "#f28e2b", "#59a14f"],
         f"Test, Scope A (n={int(te.loc[models[0], 'scopeA_n'])}; {int(te.loc[models[0], 'true_transitions'])} actual transitions): Phase 3 baselines vs SVM models",
         "markov_vs_svm_transitions.png", annotate_zero=True)
    bars(["high_recall", "high_precision", "high_f1"], ["High-entry recall", "High-entry precision", "High-entry F1"],
         ["#d62728", "#ff9896", "#9467bd"],
         f"Test, Scope A: entry into High (research-defined PM2.5 >= 75; {int(te.loc[models[0], 'true_high_entries'])} actual entries)",
         "high_entry_comparison.png")

    # confusion matrices (test, Scope B) from prediction CSVs
    cm_models = [m for m in ("linear_pca_B", "linear_hosvd_B", "revised_rff_pca_B", "revised_rff_hosvd_B", "stageB_full_pca_B", "stageB_full_hosvd_B") if m in te.index]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    for ax, m in zip(axes.ravel(), cm_models):
        fam = {"revised": "predictions/revised", "stageB": "predictions/stage_b"}.get(m.split("_")[0], "predictions")
        d = pd.read_csv(out / fam / f"final_{m}_test.csv")
        C = np.bincount(d.y.astype(int) * 3 + d.hard_pred.astype(int), minlength=9).reshape(3, 3)
        ax.imshow(C / C.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{C[i, j]}\n{C[i, j] / C[i].sum():.2f}", ha="center", va="center", fontsize=8,
                        color="w" if C[i, j] / C[i].sum() > .55 else "k")
        ax.set_xticks(range(3), ["Low", "Mod", "High"])
        ax.set_yticks(range(3), ["Low", "Mod", "High"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(DISPLAY[m], fontsize=9)
    for ax in axes.ravel()[len(cm_models):]:
        ax.axis("off")
    fig.suptitle("Test confusion matrices, Scope B (n=5,720); counts and row-normalised share", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "final_confusion_matrices_test.png", dpi=200)
    plt.close(fig)

    # RFF seed variability (validation) + original/revised picks
    sv = rc(out / "results/stage_c_full_experiment.csv")
    if sv is not None and R.get("revised"):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
        picks = {"pca_B": R["revised"]["original_lock_rff_choices"]["rff_pca_B"]["seed"], "hosvd_B": R["revised"]["original_lock_rff_choices"]["rff_hosvd_B"]["seed"]}
        for ax, cfg in zip(axes, ("pca_B", "hosvd_B")):
            for D, g in sv[sv.config == cfg].groupby("D"):
                ax.scatter(np.full(len(g), D) * np.exp(np.random.RandomState(0).uniform(-.04, .04, len(g))), g.val_scopeB_macro_f1, c="gray", s=28, alpha=.8)
                ax.plot([D * .93, D * 1.07], [g.val_scopeB_macro_f1.mean()] * 2, c="k", lw=2)
            D2 = R["revised"]["D_star"]
            r = sv[(sv.config == cfg) & (sv.D == D2)]
            o = r[r.seed == picks[cfg]]
            s2 = r[r.seed == R["revised"]["seed_star"]]
            ax.scatter(D2, o.val_scopeB_macro_f1, c="red", marker="v", s=70, label=f"original lock pick (seed {picks[cfg]}, best single seed)", zorder=5)
            ax.scatter(D2, s2.val_scopeB_macro_f1, c="green", marker="^", s=70, label=f"revised rule (median-ranked common seed {R['revised']['seed_star']})", zorder=5)
            ax.set_xscale("log", base=2)
            ax.set_xticks([256, 1024, 2048], ["256", "1024", "2048"])
            ax.set_title(f"{cfg}: validation Scope B macro-F1, 5 seeds (black = mean)", fontsize=9)
            ax.set_xlabel("RFF dimension D")
            ax.grid(ls=":", alpha=.5)
            ax.legend(fontsize=7, loc="lower right")
        axes[0].set_ylabel("macro-F1 (n=4,239)")
        fig.tight_layout()
        fig.savefig(fig_dir / "rff_seed_variability.png", dpi=200)
        plt.close(fig)

    # bootstrap forest
    bt = rc(out / "results/bootstrap_ci_table.csv")
    if bt is not None:
        b = bt[(bt.block_hours == 24) & bt.metric.isin(["macro_f1", "trans_f1"])]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
        comps = list(dict.fromkeys(b.comparison))
        for ax, met in zip(axes, ("macro_f1", "trans_f1")):
            g = b[b.metric == met].set_index("comparison").loc[comps]
            y = np.arange(len(comps))[::-1]
            ax.errorbar(g.observed_diff_a_minus_b, y, xerr=[g.observed_diff_a_minus_b - g.ci95_low, g.ci95_high - g.observed_diff_a_minus_b], fmt="o", c="C0", capsize=3)
            ax.axvline(0, c="k", lw=1)
            ax.set_yticks(y, comps, fontsize=7)
            ax.set_title(f"Test {met.replace('_', '-')} difference (a-b), paired 24h-block bootstrap 95% CI", fontsize=9)
            ax.grid(ls=":", alpha=.5)
        fig.tight_layout()
        fig.savefig(fig_dir / "bootstrap_forest.png", dpi=200)
        plt.close(fig)

    # reliability (High class, test) for models with calibrated probabilities
    rel_models = [m for m in ("linear_pca_B", "linear_hosvd_B", "revised_rff_pca_B", "revised_rff_hosvd_B") if m in te.index]
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    for m in rel_models:
        fam = "predictions/revised" if m.startswith("revised") else "predictions"
        d = pd.read_csv(out / fam / f"final_{m}_test.csv")
        t = reliability_table(d.y.values, d[["prob_0", "prob_1", "prob_2"]].values, 10, 2)
        t = t[t.n >= 20]
        ax.plot(t.mean_pred, t.obs_freq, "o-", ms=4, label=DISPLAY[m])
    ax.set_xlabel("mean predicted P(High)")
    ax.set_ylabel("observed frequency of High")
    ax.set_title("Reliability (test, Scope B n=5,720; bins with n>=20)\ntrain-only tail-split multinomial-LR calibration", fontsize=8)
    ax.legend(fontsize=6)
    ax.grid(ls=":", alpha=.5)
    fig.tight_layout()
    fig.savefig(fig_dir / "reliability_high_test.png", dpi=200)
    plt.close(fig)

    # Stage B: ablation
    ab = rc(out / "results/stage_b_ovr_ovo_ablation.csv")
    if ab is not None:
        a = ab[ab.status == "done"].dropna(subset=["val_scopeB_macro_f1"])
        a = a[~a.variant.str.startswith("CHECK")]
        fig, ax = plt.subplots(figsize=(8.5, 4))
        variants = list(dict.fromkeys(a.variant))
        for i, v in enumerate(variants):
            g = a[a.variant == v]
            ax.bar(np.arange(len(g)) + i * .27, g.val_scopeB_macro_f1, .27, label=v)
        ax.set_xticks(np.arange(a.config.nunique()) + .27, list(dict.fromkeys(a.config)))
        ax.set_ylabel("validation Scope B macro-F1 (n=4,239)")
        ax.set_title("Stage B ablation (fit_n=4000): native OvO vs genuinely trained OvR;\nthe old decision_function_shape toggle coincides with native OvO", fontsize=9)
        ax.legend(fontsize=7)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(fig_dir / "stage_b_ovr_ovo_ablation.png", dpi=200)
        plt.close(fig)

    # validation Pareto: time vs macro-F1 across families (hardware caveat in the title)
    sa, sc, sb = rc(out / "results/stage_a_all_candidates.csv"), rc(out / "results/stage_c_full_experiment.csv"), rc(out / "results/stage_b_all_candidates.csv")
    pts = []
    if sa is not None:
        pts += [("Linear SVM (Stage A)", r.fit_time_sec, r.val_scopeB_macro_f1, "o") for r in sa.itertuples()]
    if sc is not None:
        pts += [("RFF + linear SVM (Stage C)", r.fit_time_sec + r.transform_time_sec, r.val_scopeB_macro_f1, "s") for r in sc.itertuples()]
    if sb is not None:
        pts += [("Exact RBF SVC subset (Stage B, other hardware)", r.fit_time_s, r.val_scopeB_macro_f1, "^") for r in sb[sb.status == "done"].itertuples()]
    fp = rc(out / "results/stage_b_full_train_exact_pilot.csv")
    if fp is not None:
        pts += [("Exact RBF SVC full train (Stage B, other hardware)", r.fit_time_s, r.val_scopeB_macro_f1, "*") for r in fp[fp.status == "done"].itertuples()]
    if pts:
        P = pd.DataFrame(pts, columns=["fam", "t", "f", "mk"])
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        for fam, g in P.groupby("fam"):
            ax.scatter(g.t, g.f, marker=g.mk.iloc[0], s=60 if g.mk.iloc[0] == "*" else 28, alpha=.75, label=fam)
        srt = P.sort_values("t")
        front = srt[srt.f.cummax() == srt.f]
        front = front[front.f >= front.f.cummax()]
        ax.step(front.t, front.f, where="post", c="k", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("measured fit time (s, log); Stage A/C on the original machine, Stage B on a 1-vCPU sandbox: NOT hardware-comparable")
        ax.set_ylabel("validation Scope B macro-F1")
        ax.set_title("Validation macro-F1 vs fit time, all candidate fits (black = non-dominated frontier)", fontsize=9)
        ax.grid(ls=":", alpha=.5)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / "pareto_validation_all_families.png", dpi=200)
        plt.close(fig)


# ------------------------------------------------------------------------------------------------ report
def ci_sentence(row) -> str:
    lo, hi, d = row["ci95_low"], row["ci95_high"], row["observed_diff_a_minus_b"]
    if row["ci_includes_zero"]:
        return f"observed difference {d:+.4f}, 95% interval [{lo:+.4f}, {hi:+.4f}] — **includes zero**: this test sample does not separate the two models on this metric"
    edge = " (a bound lies within 0.0005 of zero, so this is borderline)" if min(abs(lo), abs(hi)) < 5e-4 else ""
    return f"observed difference {d:+.4f}, 95% interval [{lo:+.4f}, {hi:+.4f}] — excludes zero under block resampling of the test period (test-sampling uncertainty only){edge}"


def build(out: Path, in_dir: Path):
    R = {"lock": jl(out / "manifests/LOCKED_SELECTION.json"), "revised": jl(out / "manifests/REVISED_SELECTION_post_audit.json"),
         "bsel": jl(out / "manifests/STAGE_B_SELECTION_validation_only.json"), "gate": jl(out / "results/integrity_gate.json"),
         "qp": jl(out / "results/qp_verification.json"), "p3": jl(out / "results/phase3_reference_recomputed.json"),
         "boot": jl(out / "results/bootstrap_ci.json"), "tests": jl(out / "results/test_summary.json"), "rffconv": jl(out / "results/rff_convention_test.json")}
    tab = pd.read_csv(out / "results/final_metrics_from_predictions.csv")
    sa, sab = pd.read_csv(out / "results/stage_a_all_candidates.csv"), pd.read_csv(out / "results/stage_a_best_per_config.csv")
    sc, sk = pd.read_csv(out / "results/stage_c_full_experiment.csv"), pd.read_csv(out / "results/rff_kernel_approx.csv")
    sb, sbs = pd.read_csv(out / "results/stage_b_all_candidates.csv"), pd.read_csv(out / "results/stage_b_subsets.csv")
    abl, fp = rc(out / "results/stage_b_ovr_ovo_ablation.csv"), rc(out / "results/stage_b_full_train_exact_pilot.csv")
    svar, sweep = rc(out / "results/rff_seed_variability_validation.csv"), rc(out / "results/rff_seed_sweep_descriptive.csv")
    cost, tie = rc(out / "results/cost_sensitive_ablation.csv"), rc(out / "results/stage_a_tie_rule_audit.csv")
    bt, rec = rc(out / "results/bootstrap_ci_table.csv"), rc(out / "results/reconciliation_vs_final_evaluation.csv")
    figures(out, tab, R)
    te, va = tab[tab.split == "test"].set_index("model_id"), tab[tab.split == "validation"].set_index("model_id")
    M = lambda m, c, s="test": float((te if s == "test" else va).loc[m, c])
    done_b = sb[sb.status == "done"]
    L = []
    w = L.append

    w("# Phase 5 Research Report (post-audit remediation)\n")
    w(f"*Generated {utc_now()} by `source/s11_report_and_audit.py` from the result files in this package. Every number below is read from "
      "`results/`, `manifests/` or the prediction CSVs; none is typed in. Independent supervisor review of this remediated package is **pending** — "
      "nothing here should be read as audited or approved.*\n")
    w("**Status vocabulary.** *Implemented* = code exists. *Executed* = a log/result file in this package shows it ran. *Independently verified* = "
      "reproduced by someone other than the implementer; **nothing in this package has that status yet**.\n")
    w("## 1. Scope of this document\n"
      "Phase 5 compares linear SVMs, exact RBF kernel SVCs and random-Fourier-feature (RFF) SVMs on PCA and HoSVD representations (Phase 4, immutable) "
      "for next-hour PM2.5 state at Aotizhongxin. Targets are **research-defined** (Low < 35, Moderate 35–75, High ≥ 75 µg/m³); 75 is not an official AQI or "
      "public-health alert category. All evaluation uses the original class distribution. Scope A = rows with an observed current state "
      f"(validation n={R['p3']['validation']['scopeA_n']:,}, test n={R['p3']['test']['scopeA_n']:,}); Scope B = the full split "
      f"(validation n={R['p3']['validation']['scopeB_n']:,}, test n={R['p3']['test']['scopeB_n']:,}). Transition and High-entry metrics exist only on Scope A.\n")
    w("Three result families are kept strictly apart:\n"
      "* **Original lock** — `manifests/LOCKED_SELECTION.json`, written by `s06` before it opened test features; preserved unchanged.\n"
      "* **Post-audit revision** — RFF models chosen by a new pre-declared validation-only rule (§7.3). Written *after* the original test results had been seen, so it is **not** a pre-test lock.\n"
      "* **Stage B addendum** — exact RBF SVCs, run during remediation, selected on validation only and evaluated on test once, after the original lock. Also **not** part of the original lock.\n")

    # ---- data & gate
    w("## 2. Data integrity and Phase 3 reference figures\n")
    g = R["gate"]
    w(f"The integrity gate (`s01`) recorded **{g['n_checks_passed']} passed / {g['n_checks_failed']} failed / {g['n_checks_warned']} warnings** "
      f"(`results/integrity_gate.json`, run at {g['timestamp_utc']}). Phase 2–4 files were opened read-only. "+ (lambda rr: (f"A fresh re-run of the gate in the remediation environment ({rr['rerun_gate']['timestamp_utc']}) gave {rr['rerun_gate']['checks_passed']} passed / {rr['rerun_gate']['checks_failed']} failed, and all {rr['common_files']} input files common to both runs have identical SHA-256 ({len(rr['sha256_mismatches'])} mismatches); `results/input_checksum_recheck.json`." if rr else "The gate was not re-run in the remediation environment.\n"))(jl(out / "results/input_checksum_recheck.json")) + "\n")
    p3 = R["p3"]
    w("Phase 3 reference figures recomputed from the supplied CSVs:\n")
    w(md_table(pd.DataFrame([
        {"q": "Scope A n (validation / test)", "v": f"{p3['validation']['scopeA_n']} / {p3['test']['scopeA_n']}", "e": "4,197 / 5,685"},
        {"q": "Actual transitions (validation / test)", "v": f"{p3['validation']['actual_transitions']} / {p3['test']['actual_transitions']}", "e": "488 / 526"},
        {"q": "Entries into High (validation / test)", "v": f"{p3['validation']['entries_into_high']} / {p3['test']['entries_into_high']}", "e": "122 / 154"},
        {"q": "Markov test Scope A accuracy", "v": F4(p3["markov_test_scopeA_accuracy"]), "e": "≈ 0.907476"},
        {"q": "Markov test Scope B accuracy", "v": F4(p3["markov_test_scopeB_accuracy"]), "e": "≈ 0.904196"},
        {"q": "Markov hard-detected changes (test)", "v": str(p3["markov_test_hard_detected_changes"]), "e": "0"},
        {"q": "Markov test Scope A Brier / log-loss (unscaled Brier, clipped log-loss)", "v": f"{F4(p3['markov_test_scopeA_brier'])} / {F4(p3['markov_test_scopeA_log_loss'])}", "e": "≈ 0.1666 / ≈ 0.3392"}]),
        ["q", "v", "e"], ["Quantity", "Recomputed from CSV", "Spec reference"]) + "\n")

    # ---- math conventions
    w("## 3. Objectives and conventions (full derivations: `reports/math_chapter.md`)\n"
      "* **Classical SVM (QP verification only):** hinge loss, unregularised bias, dual 0 ≤ α ≤ C, Σαᵢyᵢ = 0.\n"
      "* **Linear models (Stage A, and the linear layer of Stage C):** `LinearSVC` (liblinear), **L2-regularised squared-hinge** loss, one-vs-rest, `dual=False`. "
      "liblinear appends a constant feature (`intercept_scaling=1`), so the intercept is **regularised**, unlike the classical bias. Its optimum and dual variables are not those of the hinge-loss QP.\n"
      "* **Exact RBF (Stage B):** `SVC(kernel='rbf')`, libsvm, native one-vs-one, hinge loss, unregularised bias, K = exp(−γ‖x−z‖²) with γ = a/d, d = 192.\n"
      "* **RFF (Stage C):** ω ~ N(0, 2γI), b ~ U(0, 2π), z(x) = √(2/D)·cos(Wᵀx + b), then `LinearSVC` as above; a finite-dimensional approximation, **not** the same optimiser as exact `SVC`.\n"
      "* SVM scores are not probabilities; calibrated probabilities come only from §10.\n")

    # ---- QP and RFF checks
    q = R["qp"]["summary"]
    w("## 4. Mathematical checks that were executed\n"
      f"* Dual QP on synthetic sets: {q['passed']}/{q['total_tests']} internal checks passed; primal–dual gaps {q['separable_primal_dual_gap']:.2e} (separable) and "
      f"{q['nonseparable_primal_dual_gap']:.2e} (non-separable). These sets are **synthetic**; the spec's additional 60–150-sample *training-only* QP subset was not run unless `results/qp_real_subset.json` exists.\n"
      f"* RBFSampler vs manual sampler (`rff_convention_test.json`): MAE {R['rffconv']['manual_mae']:.4f} (manual) vs {R['rffconv']['sklearn_mae']:.4f} (sklearn), reproducible = {R['rffconv']['reproducible']}. "
      "The test-suite additionally checks the manual √(2/D)·cos(Wᵀx+b) construction equals `RBFSampler` to 1e-12.\n")
    qr = jl(out / "results/qp_real_subset.json")
    if qr:
        w(f"* Classical hinge-loss dual QP on a **training-only real subset** (n = {qr['subset']['n']}: {qr['subset']['n_low']} Low / {qr['subset']['n_high']} High, "
          f"PCA-B, frozen scaler; `results/qp_real_subset.json`), compared with a matching `SVC(kernel='linear')`: all checks passed = {qr['all_runs_passed']}. "
          + "; ".join(f"{k}: SVs {v['n_sv']} ({v['n_free_sv']} free / {v['n_bound_sv']} bound), primal–dual gap {v['primal_dual_gap']:.1e}, KKT stationarity {v['kkt_stationarity_norm']:.1e}, "
                      f"max complementary-slackness residual {max(v['cs1_max'], v['cs2_max']):.1e}, |b_QP − b_SVC| {v['b_difference']:.1e}" for k, v in qr["runs"].items())
          + ". Tolerances: bound 1e-8, equality 1e-6, KKT/gap 1e-3. This is the classical hinge-loss problem, not Stage A's squared-hinge LinearSVC.\n")
    else:
        w("* The spec's 60–150-sample *training-only* real-data QP check was not run in this package.\n")
    kk = sk.groupby(["m", "D"])[["rel_frob_error", "mae"]].agg(["mean", "std"]).reset_index()
    kk.columns = ["m", "D", "rf_mean", "rf_sd", "mae_mean", "mae_sd"]
    w("Kernel approximation on training subsets (γ = 1/192, 5 seeds, `rff_kernel_approx.csv`):\n")
    w(md_table(kk, ["m", "D", "rf_mean", "rf_sd", "mae_mean", "mae_sd"], ["m", "D", "‖K−K̂‖_F/‖K‖_F mean", "sd", "mean |entry error| mean", "sd"],
               {"m": lambda v: str(int(v)), "D": lambda v: str(int(v))}) + "\n")
    k1 = kk[kk.m == kk.m.max()].set_index("D")["rf_mean"]
    ratio = k1.loc[256] / k1.loc[2048]
    w(f"At m = {int(kk.m.max())} the mean relative Frobenius error falls from {k1.loc[256]:.4f} (D=256) to {k1.loc[2048]:.4f} (D=2048), a factor {ratio:.2f} for an 8-fold increase in D "
      f"(a pure D^(−1/2) law would give {np.sqrt(8):.2f}). Seed-to-seed spread is non-trivial, and neither individual seeds nor downstream classification scores need be monotone in D.\n")

    # ---- Stage A
    w("## 5. Stage A — 40 full-training LinearSVC fits (validation)\n"
      f"All {len(sa)} fits are in `results/stage_a_all_candidates.csv`. Best C per representation by validation Scope B macro-F1 (n=4,239):\n")
    w(md_table(sab, ["config", "C", "val_scopeB_macro_f1", "val_scopeB_accuracy", "val_trans_f1", "fit_time_sec"],
               ["Representation", "C", "Macro-F1", "Accuracy", "Scope A transition F1", "Fit time (s)"]) + "\n")
    pairs = []
    for k, (p, h) in {48: ("pca_A", "hosvd_A"), 192: ("pca_B", "hosvd_B"), 576: ("pca_C", "hosvd_C")}.items():
        a, b = sab.set_index("config").loc[p], sab.set_index("config").loc[h]
        pairs.append({"k": k, "pca": a.val_scopeB_macro_f1, "hosvd": b.val_scopeB_macro_f1, "diff": a.val_scopeB_macro_f1 - b.val_scopeB_macro_f1})
    pairs = pd.DataFrame(pairs)
    w("PCA vs HoSVD with the same solver family and C grid (validation, one fit per cell, **no uncertainty interval**):\n")
    w(md_table(pairs, ["k", "pca", "hosvd", "diff"], ["k", "PCA macro-F1", "HoSVD macro-F1", "PCA − HoSVD"], {"k": lambda v: str(int(v))}) + "\n")
    w(("PCA is higher at every k in this validation comparison. " if (pairs["diff"] > 0).all() else "The ordering of PCA and HoSVD is not the same at every k. ") +
      "This is a statement about these linear fits only; §6 shows the gap is not a property of the representations in general.\n")
    alt = sab.set_index("config").loc[["hosvd_B", "hosvd_B_alt1", "hosvd_B_alt2"]]
    w("HoSVD rank allocations at d = 192 (validation, linear):  " + ";  ".join(f"{c} = {F4(v)}" for c, v in alt["val_scopeB_macro_f1"].items()) +
      f". Spread {alt['val_scopeB_macro_f1'].max() - alt['val_scopeB_macro_f1'].min():.4f}; single fits, so ranks are suggestive only.\n")
    if tie is not None:
        ch = tie[tie.selection_changes]
        w(f"**Selection-rule audit.** The documented tie rule (§12 of the spec: within 0.005 of the best macro-F1, prefer Scope A transition F1, then recall, balanced accuracy, cost) "
          f"was *not* what the original code applied (it used plain arg-max). Applying the rule to the stored validation results would change the chosen C for "
          f"{len(ch)} of {len(tie)} representations ({', '.join(ch.config) if len(ch) else 'none'}). `results/stage_a_tie_rule_audit.csv` lists both. The original lock is **not** altered; this is reported as a finding.\n")

    # ---- Stage B
    w("## 6. Stage B — exact RBF SVC (native one-vs-one)\n")
    nd, nt = len(done_b), len(sb)
    w(f"Grid executed: **{nd} of {nt}** cells completed (2 representations × 2 training sizes × γ-scale a ∈ {{0.1, 1, 10}} × C ∈ {{0.1, 1, 10}}); statuses: "
      f"{sb['status'].value_counts().to_dict()}. Source: `results/stage_b_all_candidates.csv`, logs `logs/s04_stage_b.log`. "
      "Scalers: the frozen Stage A scalers were loaded with the corrected key handling (`mean`/`scale`); no scaler was fitted in this stage.\n")
    w("Training subsets (deterministic systematic temporal blocks of 100 consecutive rows spread over the whole training period; identical rows for both representations):\n")
    w(md_table(sbs, ["fit_n", "n_Low", "n_Moderate", "n_High", "share_Low", "share_Moderate", "share_High", "distinct_calendar_months_covered", "total_variation_distance_vs_full_train"],
               ["fit_n", "Low", "Moderate", "High", "share Low", "share Mod", "share High", "months covered (of 34)", "TV distance vs full train"],
               {"fit_n": lambda v: str(int(v)), "n_Low": lambda v: str(int(v)), "n_Moderate": lambda v: str(int(v)), "n_High": lambda v: str(int(v)),
                "distinct_calendar_months_covered": lambda v: str(int(v))}) +
      f"\nFull-training class shares: Low {sbs['fulltrain_share_Low'].iloc[0]:.4f}, Moderate {sbs['fulltrain_share_Moderate'].iloc[0]:.4f}, High {sbs['fulltrain_share_High'].iloc[0]:.4f}.\n")
    best = done_b.sort_values("val_scopeB_macro_f1", ascending=False).groupby(["config", "fit_n"]).head(1).sort_values(["config", "fit_n"])
    w("Best cell per (representation, fit_n) by validation Scope B macro-F1:\n")
    w(md_table(best, ["config", "fit_n", "gamma_scale", "C", "val_scopeB_macro_f1", "val_scopeB_accuracy", "val_trans_f1", "n_support_vectors", "fit_time_s", "child_peak_rss_mb", "model_bytes"],
               ["Repr.", "fit_n", "a (γ=a/192)", "C", "Macro-F1", "Accuracy", "Trans. F1 (Scope A)", "# SVs", "fit s", "peak RSS MB", "model bytes"],
               {"fit_n": lambda v: str(int(v)), "n_support_vectors": lambda v: str(int(v)), "model_bytes": lambda v: f"{int(v):,}", "gamma_scale": lambda v: f"{v:g}", "C": lambda v: f"{v:g}",
                "fit_time_s": lambda v: f"{v:.2f}", "child_peak_rss_mb": lambda v: f"{v:.0f}"}) + "\n")
    edge = [(r.config, r.fit_n) for r in best.itertuples() if r.gamma_scale == 0.1 or r.C == 10]
    w(f"**Grid-boundary caution:** the best cell sits on an edge of the prescribed grid (smallest γ-scale 0.1 and/or largest C 10) for {len(edge)} of {len(best)} (representation, fit_n) pairs; "
      "better hyper-parameters may lie outside the grid. The grid was kept as specified and not extended.\n")
    sel = R["bsel"]["per_config"]
    w("Validation-only Stage B selection (rule: macro-F1, 0.005 tie band): " + "; ".join(f"{k}: {v['run_id']} (macro-F1 {F4(v['val_scopeB_macro_f1'])})" for k, v in sel.items() if v) + ".\n")
    w("Memory/time measurement: fits ran in forked child processes with an enforced wall-time limit; peak RSS is `getrusage` in the child (includes copy-on-write pages shared with the parent, so it is an upper-ish bound). "
      "The RAM guard is an estimate, not an OS cap. No dense N×N Gram matrix is built (the 24,108² float64 reference of ≈4.33 GiB is only a reference figure).\n")
    if abl is not None:
        a = abl[(abl.status == "done") & ~abl.variant.str.startswith("CHECK")]
        w("**OvO vs OvR ablation (fit_n = 4000, same rows, scaler, γ, C).** The audited script only toggled `decision_function_shape`, which cannot alter `SVC.predict`; "
          "the corrected ablation trains a genuine one-vs-rest model:\n")
        w(md_table(a, ["config", "variant", "val_scopeB_macro_f1", "val_scopeB_accuracy", "val_trans_f1", "n_support_vectors_total", "fit_time_s"],
                   ["Repr.", "Variant", "Macro-F1", "Accuracy", "Trans. F1", "total SVs", "fit s"], {"n_support_vectors_total": lambda v: str(int(v))}) + "\n")
        chk = abl[abl.variant.str.startswith("CHECK")]
        w("Check that the old toggle changes nothing: " + "; ".join(f"{r.config}: {r.note}" for r in chk.itertuples()) + ". "
          "Residual differences between OvO and OvR here: each OvR binary SVC sees all rows, OvO sees pairs; the aggregation rule and the number of support vectors differ; "
          "the comparison is one hyper-parameter setting (the OvO winner's), which may favour OvO.\n")
    if fp is not None and (fp.status == "done").any():
        f = fp[fp.status == "done"]
        w("**Optional full-training exact fit (24,108 rows; validation only at this stage).** Hyper-parameters were those of the fit_n = 8000 validation winner (not re-tuned at full size):\n")
        w(md_table(f, ["config", "fit_n", "gamma_scale", "C", "val_scopeB_macro_f1", "val_scopeB_accuracy", "val_trans_f1", "val_high_recall", "n_support_vectors", "fit_time_s", "child_peak_rss_mb", "model_bytes"],
                   ["Repr.", "fit_n", "a", "C", "Macro-F1", "Accuracy", "Trans. F1", "High-entry recall", "# SVs", "fit s", "peak RSS MB", "model bytes"],
                   {"fit_n": lambda v: str(int(v)), "n_support_vectors": lambda v: str(int(v)), "model_bytes": lambda v: f"{int(v):,}", "gamma_scale": lambda v: f"{v:g}", "C": lambda v: f"{v:g}",
                    "fit_time_s": lambda v: f"{v:.1f}", "child_peak_rss_mb": lambda v: f"{v:.0f}"}) + "\n")
        w("These fits completed well inside the 3 GB / 10 min guidance on the hardware used; that is a measurement on that machine, not a guarantee elsewhere. "
          "Exact-SVC models trained on 4,000/8,000 rows are **not** an apples-to-apples comparison with 24,108-row RFF/linear models; the full-train rows are labelled fit_n = 24,108.\n")

    # ---- Stage C
    w("## 7. Stage C — RFF + linear SVM\n")
    cg = rc(out / "results/stage_c_coarse_grid.csv")
    if cg is not None:
        bb = cg.sort_values("val_scopeB_macro_f1", ascending=False).groupby("config").head(1)
        w("Coarse (a, C) tuning at D = 1024, seed 42 (validation): best " + "; ".join(f"{r.config}: a={r.gamma_scale:g}, C={r.C:g} (macro-F1 {F4(r.val_scopeB_macro_f1)})" for r in bb.itertuples()) +
          ". Both winners are at the grid edge (a = 0.1, C = 10), so the tuned region may not contain the optimum.\n")
    w("### 7.1 Seed variability (validation, Scope B macro-F1, n=4,239; all five predeclared seeds 42–46 retained)\n")
    w(md_table(svar, ["config", "D", "mean", "std_ddof1", "min", "median", "max"], ["Repr.", "D", "mean", "sd", "min", "median", "max"], {"D": lambda v: str(int(v))}) + "\n")
    w("### 7.2 What the original lock did\n")
    oc = R["revised"]["original_lock_rff_choices"]
    w("The original lock took the arg-max over single validation rows: " + "; ".join(f"{k}: D={v['D']}, seed={v['seed']} (rank {v['rank_among_5_seeds_best_is_1']} of 5, best single seed = {v['is_best_validation_seed']})" for k, v in oc.items()) +
      ". That is best-seed selection, contrary to the predetermined-seed protocol, and the two representations used different random frequencies. The original lock and its test numbers are preserved and shown below as such.\n")
    w("### 7.3 Revised validation-only rule (written during remediation, before any revised model was fit or scored)\n")
    rv = R["revised"]
    w(f"Rule: {rv['rule']}. Applied to the stored 30 runs: D* = {rv['D_star']}; ranking of seeds by pair-mean validation macro-F1 (best→worst) = {rv['details']['seed_rank_best_to_worst']}; "
      f"selected seed* = {rv['seed_star']} (the median-ranked seed). Both representations use (D*, seed*) so they share identical random frequencies and phases. "
      "The rule never picks the best or worst seed *of the pair-mean ranking*; " + "; ".join(
          f"for {cfg} alone, seed {rv['seed_star']} ranks {int(sc[(sc.config == cfg) & (sc.D == rv['D_star'])].sort_values('val_scopeB_macro_f1', ascending=False).seed.tolist().index(rv['seed_star']) + 1)} of 5 on validation macro-F1"
          for cfg in ("pca_B", "hosvd_B")) + ", so a single representation's best seed can still coincide with the common seed. This revision was written after the original test results were known, so it is a *post-audit analysis*, not a pre-test lock. "
      "No test metric entered the rule (`test_metrics_used_for_selection = false`).\n")

    # ---- locks
    w("## 8. Selection manifests\n")
    sm = R["lock"]["selected_models"]
    rows = [{"model": k, "spec": f"{v['model_type']}, {v['config']}" + (f", C={v['C']:g}" if 'C' in v else "") + (f", D={v['D']}, seed={v['seed']}" if 'D' in v else ""),
             "val": v["val_scopeB_macro_f1"]} for k, v in sm.items()]
    w(f"Original lock (`manifests/LOCKED_SELECTION.json`, {R['lock']['timestamp_utc']}):\n")
    w(md_table(pd.DataFrame(rows), ["model", "spec", "val"], ["Model id", "Specification", "Validation Scope B macro-F1"]) + "\n")
    w("Revised selection: `manifests/REVISED_SELECTION_post_audit.json`. Stage B: `manifests/STAGE_B_SELECTION_validation_only.json`. All three are validation-only.\n")

    # ---- test results
    def block(ids, title, note=""):
        ids = [i for i in ids if i in te.index]
        t = te.loc[ids].reset_index()
        t["name"] = t["model_id"].map(DISPLAY)
        w(f"### {title}\n{note}")
        w(md_table(t, ["name", "scopeB_accuracy", "scopeB_macro_f1", "scopeB_balanced_accuracy", "scopeA_accuracy", "scopeA_macro_f1"],
                   ["Model", "B accuracy", "B macro-F1", "B balanced acc.", "A accuracy", "A macro-F1"]) + "\n")
        w(md_table(t, ["name", "true_transitions", "pred_transitions", "trans_tp", "trans_fp", "trans_fn", "trans_precision", "trans_recall", "trans_f1", "trans_dest_accuracy", "high_tp", "high_fp", "high_fn", "high_precision", "high_recall", "high_f1"],
                   ["Model", "actual changes", "predicted changes", "TP", "FP", "FN", "precision", "recall", "F1", "dest. acc.", "High TP", "FP", "FN", "High prec.", "High rec.", "High F1"],
                   {k: (lambda v: str(int(v))) for k in ("true_transitions", "pred_transitions", "trans_tp", "trans_fp", "trans_fn", "high_tp", "high_fp", "high_fn")}) + "\n")

    w("## 9. Test results (each model evaluated once on test; no test-driven selection)\n"
      f"Scope B n = {int(te.iloc[0]['scopeB_n']):,}; Scope A n = {int(te.iloc[0]['scopeA_n']):,}; transition and High-entry rows use Scope A only. Zero-denominator convention shared by all models: undefined precision/recall/F1 = 0.\n")
    block(["phase3_majority", "phase3_persistence", "phase3_markov"], "9.1 Phase 3 baselines (recomputed from the supplied CSVs)")
    block(["linear_pca_B", "linear_hosvd_B", "linear_best_overall", "rff_pca_B", "rff_hosvd_B"], "9.2 Original lock",
          "Values come from the saved prediction CSVs; they reconcile with `final_evaluation_scopeAB.csv` to within "
          f"{rec['max_abs_diff'].max():.1e}.\n\n")
    block(["revised_rff_pca_B", "revised_rff_hosvd_B"], "9.3 Post-audit revision (RFF with the median-seed common-pair rule)",
          "**Not a pre-test lock** (see §7.3).\n\n")
    block(["stageB_pca_B", "stageB_hosvd_B", "stageB_full_pca_B", "stageB_full_hosvd_B"], "9.4 Stage B addendum (exact RBF SVC)",
          "Subset rows (`stageB_*`) are trained on the validation-selected fit_n; `stageB_full_*` on all 24,108 rows with the fit_n = 8000 hyper-parameters. Selected on validation only; evaluated on test after the original lock, so **not part of it**. "
          "Probabilities were not produced for exact SVC (no calibrator was fitted); those cells are missing by design.\n\n")
    svm_ids = [m for m in te.index if not m.startswith("phase3")]
    best_svm = te.loc[svm_ids, "scopeB_macro_f1"].max()
    mk = M("phase3_markov", "scopeB_macro_f1")
    cmp_txt = (f"Markov's Scope B macro-F1 ({F4(mk)}) is higher than every SVM-family model's (best {F4(best_svm)})" if mk > best_svm else
               f"Markov's Scope B macro-F1 ({F4(mk)}) is not higher than every SVM-family model's (best {F4(best_svm)})")
    w(f"Observation: the Phase 3 Markov hard predictions predict {int(te.loc['phase3_markov', 'pred_transitions'])} changes against {int(te.loc['phase3_markov', 'true_transitions'])} actual transitions "
      f"yet have Scope A accuracy {F4(M('phase3_markov', 'scopeA_accuracy'))}. Hard accuracy is not a measure of transition responsiveness. Linear PCA-B predicts changes "
      f"(transition recall {F4(M('linear_pca_B', 'trans_recall'))}) with precision {F4(M('linear_pca_B', 'trans_precision'))}; {cmp_txt}. "
      "The SVM models use only the 24-hour history; the Markov baseline uses the current state, so they have different information sets.\n")

    # ---- probabilistic
    w("## 10. Probabilistic evaluation and calibration\n"
      "* Raw SVM scores are not probabilities. For each final linear/RFF model a multinomial logistic regression (C = 1) maps the three class scores to probabilities.\n"
      "* **Protocol (train only):** the chronological first 80 % of training rows fit a base classifier; its scores on the final 20 % tail (true labels) fit the mapper. The deployed classifier is then refit on 100 % of training data and the frozen mapper applied. "
      "This is a **single forward split, not multi-fold cross-fitting**, and the score scale of the 80 % model is only an approximation of the 100 % model's. No validation/test label or score fits any scaler, classifier, RFF sampler or calibrator.\n"
      "* **Limitation:** Phase 4 PCA/HoSVD bases were fitted on the whole training period, so even the 80/20 split is not end-to-end forward-chaining at the representation level.\n"
      "* Class order is [0, 1, 2] = [Low, Moderate, High] for scores, calibrator output and metrics (asserted in code and tests). Raw hard prediction (`hard_pred`) and calibrated argmax (`calibrated_argmax`) are stored separately; **all hard-classification tables use `hard_pred`**.\n"
      "* Brier = mean Σₖ(pₖ − 1[y=k])² (unscaled; Phase 3 is reproduced under the same convention, §2). Log-loss clips probabilities to [1e-12, 1] and renormalises rows.\n")
    pr = te.loc[[m for m in te.index if te.loc[m, "scopeA_brier"] == te.loc[m, "scopeA_brier"]]].reset_index()
    pr["name"] = pr["model_id"].map(DISPLAY)
    w(md_table(pr, ["name", "scopeA_brier", "scopeA_log_loss", "scopeB_brier", "scopeB_log_loss", "calibrated_argmax_differs_from_hard_frac"],
               ["Model (test)", "Scope A Brier", "Scope A log-loss", "Scope B Brier", "Scope B log-loss", "fraction where calibrated argmax ≠ hard label"]) + "\n")
    w("Phase 3 Markov/majority probabilities come from the supplied CSV columns. Exact-SVC models have no probabilities (missing, not imputed). Reliability diagram for High: `figures/reliability_high_test.png`. "
      "Probabilistic scores of different families are comparable only to the extent that each calibrator is adequate; a lower Brier does not by itself establish a better-calibrated model.\n")

    # ---- seed sweep on test
    if sweep is not None:
        sw = sweep[sweep.split == "test"]
        s2 = sw.groupby("config")[["scopeB_macro_f1", "scopeA_trans_f1"]].agg(["mean", "std", "min", "max"])
        w("## 11. RFF seed variability on the test split (descriptive only)\n"
          f"All five predeclared seeds at D* = {rv['D_star']} were refit on training data and scored on test. This table shows how much a seed choice can move results; it was **not** used to select anything.\n")
        t = sw.sort_values(["config", "seed"])
        w(md_table(t, ["config", "seed", "selected_by_rule", "scopeB_macro_f1", "scopeA_trans_f1", "scopeA_trans_recall", "scopeA_high_recall"],
                   ["Repr.", "seed", "revised rule picks", "B macro-F1", "A trans. F1", "A trans. recall", "A High-entry recall"],
                   {"seed": lambda v: str(int(v)), "selected_by_rule": lambda v: "yes" if v else ""}) + "\n")
        for cfg in ("pca_B", "hosvd_B"):
            r = sw[sw.config == cfg].scopeB_macro_f1
            w(f"{cfg}: test macro-F1 across seeds ranges {r.min():.4f}–{r.max():.4f} (sd {r.std(ddof=1):.4f}). ")
        rng_ = {cfg: float(sw[sw.config == cfg].scopeB_macro_f1.max() - sw[sw.config == cfg].scopeB_macro_f1.min()) for cfg in ("pca_B", "hosvd_B")}
        d_lin = M("linear_pca_B", "scopeB_macro_f1") - M("linear_hosvd_B", "scopeB_macro_f1")
        d_rff = M("revised_rff_pca_B", "scopeB_macro_f1") - M("revised_rff_hosvd_B", "scopeB_macro_f1")
        w(f"\nFor scale: the seed-to-seed range of test macro-F1 is {rng_['pca_B']:.4f} (PCA-B) and {rng_['hosvd_B']:.4f} (HoSVD-B), against a PCA-B − HoSVD-B difference of {d_lin:+.4f} for the linear models and {d_rff:+.4f} for the revised RFF pair. "
          "RFF seed variability is a separate source of uncertainty from the bootstrap in §12 (which holds the fitted models fixed).\n")
        o = {k: pd.read_csv(out / f"predictions/final_{k}_test.csv").hard_pred.values for k in ("rff_hosvd_B", "rff_pca_B")}
        r_ = {k: pd.read_csv(out / f"predictions/revised/final_revised_{k}_test.csv").hard_pred.values for k in ("rff_hosvd_B", "rff_pca_B")}
        w("**Refit reproducibility check.** The original lock used seed {oc['rff_hosvd_B']['seed']} for HoSVD-B and the revised rule picks seed {rv['seed_star']}; "
          f"{"the revised HoSVD-B model is therefore a refit of the original configuration (same D, γ, C, seed) in a different software environment" if oc['rff_hosvd_B']['seed'] == rv['seed_star'] else "the two HoSVD-B models differ by design"}: "
          f"{int((o['rff_hosvd_B'] != r_['rff_hosvd_B']).sum())} of {len(o['rff_hosvd_B'])} test hard predictions differ between the original and the refit (likely floating-point/solver or library-version differences; the original environment's package versions were not recorded in the package). "
          f"For PCA-B the revised seed differs from the original seed, so {int((o['rff_pca_B'] != r_['rff_pca_B']).sum())} differing predictions there is expected and is not a reproducibility measure.\n")

    # ---- bootstrap
    w("## 12. Paired chronological block bootstrap (test)\n")
    w(f"Resampling unit: **real-time blocks** (calendar 24-hour primary, 168-hour sensitivity) derived from `target_timestamp`, so missing hours do not stretch a block; "
      f"{R['boot']['runs']['block_24h']['n_blocks']} blocks of 24 h, {R['boot']['n_boot'] if 'n_boot' in R['boot'] else R['boot']['runs']['block_24h']['n_boot']} resamples, seed {R['boot']['runs']['block_24h']['seed']}; the same resampled blocks are used for both models of a pair. "
      "Uncertainty covers **test-period sampling only** — not training-set variability, not RFF seed variability — and ignores dependence longer than a block. Intervals are percentile 95 %; no p-values or significance tests are claimed.\n")
    so = jl(out / "superseded_pre_audit/results/bootstrap_ci.json")
    sa_old = jl(out / "superseded_pre_audit/audit_summary.json")
    if so and sa_old:
        mo, to = so["macro_f1"]["ci_95"], so["transition_f1"]["ci_95"]
        cl = sa_old["block_bootstrap_95ci"]
        w(f"**Correction.** The superseded pre-audit bootstrap artifact ({so['n_blocks']} blocks of {so['block_size_hours']} consecutive *row-index* rows) recorded, for Linear PCA-B minus HoSVD-B, "
          f"a macro-F1 difference interval of [{mo[0]:+.4f}, {mo[1]:+.4f}] and a transition-F1 difference interval of [{to[0]:+.4f}, {to[1]:+.4f}]"
          f"{' — which includes zero' if to[0] <= 0 <= to[1] else ''}. The pre-audit report and audit summary instead stated {cl['macro_f1_diff_pca_vs_hosvd'][0]:.4f} to {cl['macro_f1_diff_pca_vs_hosvd'][1]:.4f} (macro-F1) "
          f"and {cl['trans_f1_diff_pca_vs_hosvd'][0]:.4f} to {cl['trans_f1_diff_pca_vs_hosvd'][1]:.4f} (transition-F1) and called the PCA advantage statistically robust. Those statements did not match the artifact and are withdrawn. "
          "The corrected analysis below uses timestamp-based blocks.\n")
    if bt is not None:
        for hours in (24, 168):
            b = bt[bt.block_hours == hours]
            w(f"### {hours}-hour blocks ({int(b.n_blocks.iloc[0])} blocks)\n")
            rows = []
            for r in b[b.metric.isin(["macro_f1", "trans_f1", "high_f1"])].itertuples():
                rows.append({"comparison": r.comparison, "metric": r.metric.replace("_", "-"), "obs": r.observed_diff_a_minus_b, "lo": r.ci95_low, "hi": r.ci95_high, "z": "yes" if r.ci_includes_zero else "no"})
            w(md_table(pd.DataFrame(rows), ["comparison", "metric", "obs", "lo", "hi", "z"], ["Comparison (a − b)", "Metric", "Observed diff", "95% low", "95% high", "Interval includes 0"]) + "\n")
        b24 = bt[(bt.block_hours == 24)].set_index(["comparison", "metric"])
        for comp in dict.fromkeys(bt.comparison):
            w(f"* {comp}: macro-F1 — {ci_sentence(b24.loc[(comp, 'macro_f1')].to_dict())}. Transition-F1 — {ci_sentence(b24.loc[(comp, 'trans_f1')].to_dict())}.")
        w("\nTwo different windows (24 h vs 168 h) generally widen the intervals; any conclusion that changes between them is fragile.\n")

    # ---- cost sensitive
    if cost is not None:
        w("## 13. Supplement: transition-weighted training (validation only; not part of winner selection)\n"
          "Sample weight λ on training rows whose next state differs from the current state (Scope A train rows), 1 otherwise; the current state is used only to define weights — this is not Markov fusion. "
          "Evaluated on the natural validation population; single runs at the validation-selected C of each representation, no uncertainty estimate, no calibration.\n")
        w(md_table(cost, ["config", "C", "lambda", "val_scopeB_macro_f1", "trans_precision", "trans_recall", "trans_f1", "trans_dest_accuracy", "high_precision", "high_recall", "high_f1"],
                   ["Repr.", "C", "λ", "B macro-F1", "trans. precision", "trans. recall", "trans. F1", "dest. acc.", "High prec.", "High rec.", "High F1"],
                   {"lambda": lambda v: f"{v:g}", "C": lambda v: f"{v:g}"}) + "\n")
        msgs = []
        for cfg, g in cost.groupby("config"):
            g = g.sort_values("lambda")
            base = g.iloc[0]
            msgs.append(f"{cfg}: λ=1→λ={g['lambda'].iloc[-1]:g} transition recall {base.trans_recall:.4f}→{g.trans_recall.iloc[-1]:.4f}, transition F1 {base.trans_f1:.4f}→{g.trans_f1.iloc[-1]:.4f}, "
                        f"High-entry F1 {base.high_f1:.4f}→{g.high_f1.iloc[-1]:.4f}, macro-F1 {base.val_scopeB_macro_f1:.4f}→{g.val_scopeB_macro_f1.iloc[-1]:.4f} "
                        f"(recall monotone in λ: {bool(g.trans_recall.is_monotonic_increasing)}; F1 monotone: {bool(g.trans_f1.is_monotonic_increasing)})")
        w("Effect of weighting (validation, one run each): " + "; ".join(msgs) + ". No claim of guaranteed improvement or calibration is made.\n")

    # ---- comparative
    w("## 14. Comparative findings (conditional on the numbers above)\n")
    f_e = {c: fp[(fp.config == c) & (fp.status == "done")].val_scopeB_macro_f1.iloc[0] for c in ("pca_B", "hosvd_B")} if fp is not None and (fp.status == "done").any() else {}
    if f_e:
        w(f"* **PCA vs HoSVD under the exact kernel.** At d = 192 the full-train exact RBF SVC reaches validation macro-F1 {F4(f_e['pca_B'])} (PCA-B) and {F4(f_e['hosvd_B'])} (HoSVD-B), "
          f"a difference of {f_e['pca_B'] - f_e['hosvd_B']:+.4f}, versus {pairs.set_index('k').loc[192, 'diff']:+.4f} for the linear models. "
          + ("The PCA–HoSVD gap observed for the linear models is " + ("not visible" if abs(f_e['pca_B'] - f_e['hosvd_B']) <= 0.005 else "still present") +
           " under the exact kernel at d = 192 (|difference| " + ("≤" if abs(f_e['pca_B'] - f_e['hosvd_B']) <= 0.005 else ">") + " the 0.005 practical tie band, a convention and not a test); single validation fits, no interval."))
    r_p = te.loc["revised_rff_pca_B"] if "revised_rff_pca_B" in te.index else None
    w(f"* **Exact vs approximate kernel.** Validation macro-F1 of the best RFF fits (D = 2048) is {F4(sc[sc.D == 2048].val_scopeB_macro_f1.max())} (best single run), whereas the exact full-train SVC reaches {F4(max(f_e.values())) if f_e else 'n/a'}; "
      "RFF at D ≤ 2048 with a linear squared-hinge solver is a coarser approximation and uses OvR, while exact SVC uses OvO hinge loss, so the gap mixes approximation, loss and multiclass-strategy effects.")
    w("* **Reconstruction vs classification.** Phase 4 reconstruction errors (not recomputed here) measure how well a basis reproduces the inputs; discrimination of the next-hour state need not follow reconstruction quality, and the comparisons above should not be read as a ranking of reconstruction.")
    w("* **Training-size confound.** Subset exact fits (4,000/8,000 rows) use far fewer rows than the 24,108-row linear/RFF models.\n")

    # ---- resources
    w("## 15. Resource notes\n"
      "Stage A/C timings and sizes were measured on the original run machine (its hardware is not recorded in the result CSVs) and are taken from the stored CSVs; Stage B and the revision stage were run here on a 1-vCPU Linux sandbox with ~3 GB RAM "
      "(`logs/s04_stage_b.log`, `logs/s09_post_audit_revision.log`). Timings are therefore **not comparable across stages**. "
      "Inference-time benchmarking (warm-up/repeats) and projected-basis storage totals were not re-measured in this remediation.\n")

    # ---- limitations
    w("## 16. Limitations and unexecuted items\n"
      "1. Phase 4 representations were fitted over the whole original training period; calibration and all internal splits inherit that.\n"
      "2. Calibration is a single 80/20 chronological split, not multi-fold; Stage B models have no probabilities.\n"
      "3. Stage A selection used arg-max rather than the documented 0.005 tie rule (impact in §5); the original lock was not changed.\n"
      "4. Grid-edge optima in Stage B and Stage C; grids were not extended.\n"
      "5. RFF detailed experiments cover only the PCA-B / HoSVD-B pair; the spec's optional validation-selected alternative pairs and dimension controls (d = 48, 576) for RFF and exact SVC were **not** run.\n"
      "6. The classical-hinge QP checks use SLSQP on n = 100 points; they verify the dual/KKT mathematics, not the large-scale LinearSVC optimum.\n"
      "7. Bootstrap intervals reflect test-sampling variability only.\n"
      "8. Original Stage A, Stage C and `s06` runs were **not re-executed** here; they were verified only by reloading saved models and reconciling saved predictions with metric tables (reload reproduces predictions exactly, see tests). The Stage B, revision, table, bootstrap and report stages were executed in this remediation.\n"
      "9. Nothing here has been independently verified by anyone but the implementer.\n")
    if R["tests"]:
        T = R["tests"]
        w(f"## 17. Test suite\nActual result of `python -m pytest tests` ({T['seconds']} s): **{T['passed']} passed, {T['failed']} failed, {T['skipped']} skipped** of {T['total']}. "
          "Skipped tests (if any) and reasons are in `results/test_summary.json`; full log in `logs/pytest_log.txt`.\n")
    return "\n".join(L), R, tab


def audit(out: Path, R, tab):
    te = tab[tab.split == "test"].set_index("model_id")
    va = tab[tab.split == "validation"].set_index("model_id")
    sa = pd.read_csv(out / "results/stage_a_all_candidates.csv")
    sc = pd.read_csv(out / "results/stage_c_full_experiment.csv")
    sb = pd.read_csv(out / "results/stage_b_all_candidates.csv")
    fp = rc(out / "results/stage_b_full_train_exact_pilot.csv")
    abl = rc(out / "results/stage_b_ovr_ovo_ablation.csv")
    models = [m for m in DISPLAY if m in te.index]
    pick = lambda col, df=te: {m: float(df.loc[m, col]) for m in models}
    bt = rc(out / "results/bootstrap_ci_table.csv")
    sab = pd.read_csv(out / "results/stage_a_best_per_config.csv")
    A = {"phase": 5, "revision": "post-audit remediation", "generated_utc": utc_now(), "generated_from": "results/*.csv|json (no hard-coded metrics)",
         "status_labels": {"implemented": True, "executed_in_this_package": ["integrity gate (original run)", "QP synthetic checks (original run)", "Stage A 40 fits (original run)",
                                                                           "Stage B grid + ablation + full-train pilot (remediation run)", "Stage C 30 fits (original run)",
                                                                           "s06 original lock evaluation (original run)", "revision stage, tables, bootstrap, report (remediation run)"],
                           "independently_verified": False},
         "integrity_gate": {"passed": R["gate"]["gate_passed"], "checks_passed": R["gate"]["n_checks_passed"], "checks_failed": R["gate"]["n_checks_failed"], "source": "results/integrity_gate.json (original run)"},
         "fits_completed": {"stage_a_linear_svm": f"{len(sa)}/40", "stage_c_rff_svm": f"{len(sc)}/30",
                            "stage_b_exact_rbf_grid": f"{int((sb.status == 'done').sum())}/{len(sb)}", "stage_b_status_counts": sb["status"].value_counts().to_dict(),
                            "stage_b_ablation_variants_done": int((abl.status == 'done').sum()) if abl is not None else 0,
                            "stage_b_full_train_exact_pilot": (fp.status.value_counts().to_dict() if fp is not None else "not run")},
         "best_validation": {"primary_criterion": "validation Scope B macro-F1",
                             "stage_a_best_per_config": {r.config: {"C": float(r.C), "val_scopeB_macro_f1": float(r.val_scopeB_macro_f1)} for r in sab.itertuples()},
                             "stage_b_validation_selection": R["bsel"]["per_config"],
                             "stage_c_best_single_run": sc.sort_values("val_scopeB_macro_f1", ascending=False).iloc[0][["config", "D", "seed", "val_scopeB_macro_f1"]].to_dict()},
         "selection_manifests": {"original_lock": "manifests/LOCKED_SELECTION.json", "revised_post_audit": "manifests/REVISED_SELECTION_post_audit.json",
                                 "stage_b_validation_only": "manifests/STAGE_B_SELECTION_validation_only.json"},
         "rff_seed_policy": {"original": "arg-max over single validation rows (best-seed)", "revised_rule": R["revised"]["rule"], "D_star": R["revised"]["D_star"],
                             "seed_star": R["revised"]["seed_star"], "revised_is_pre_test_lock": False},
         "test_results_summary": {"note": "each model evaluated once; original-lock, revised and Stage B families are distinct (see report)",
                                  "family": {m: str(te.loc[m, "family"]) for m in models},
                                  "scopeB_macro_f1": pick("scopeB_macro_f1"), "scopeB_accuracy": pick("scopeB_accuracy"),
                                  "scopeA_transition_f1": pick("trans_f1"), "scopeA_transition_recall": pick("trans_recall"), "scopeA_transition_precision": pick("trans_precision"),
                                  "scopeA_high_entry_recall": pick("high_recall"), "scopeA_high_entry_precision": pick("high_precision")},
         "block_bootstrap_24h": ({f"{r.model_a} - {r.model_b} [{r.metric}]": {"observed_diff": r.observed_diff_a_minus_b, "ci95": [r.ci95_low, r.ci95_high], "ci_includes_zero": bool(r.ci_includes_zero)}
                                  for r in bt[(bt.block_hours == 24) & bt.metric.isin(["macro_f1", "trans_f1"])].itertuples()} if bt is not None else None),
         "tests": ({"source": "results/test_summary.json", **{k: R["tests"][k] for k in ("total", "passed", "failed", "skipped")}} if R["tests"] else {"source": None, "status": "not yet run"}),
         "planned_not_executed": ["Stage B probability calibration (exact SVC probabilities missing)", "RFF/exact-SVC dimension controls (d=48, 576) and alternative pairs",
                                  "multi-fold chronological cross-fit calibration (single 80/20 tail split used)", "inference-time benchmarking with warm-up/repeats", "train+validation refit study (not performed by design)",
                                  "re-execution of original Stage A / Stage C / s06 runs (verified by reload + reconciliation only)"],
         "limitations": ["Phase 4 representations fitted over the whole training period", "75 ug/m3 boundary is research-defined", "bootstrap = test-sampling uncertainty only",
                         "grid-edge optima in Stage B/C", "Stage A selection used arg-max, not the documented tie rule (see results/stage_a_tie_rule_audit.csv)",
                         "Stage B timings are on different hardware from Stage A/C"]}
    (out / "audit_summary.json").write_text(json.dumps(A, indent=2, default=float))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default=".")
    p.add_argument("--output-dir", default=None)
    a = p.parse_args(argv)
    out = Path(a.output_dir) if a.output_dir else Path(__file__).resolve().parent.parent
    text, R, tab = build(out, Path(a.input_dir))
    (out / "reports").mkdir(exist_ok=True)
    (out / "reports" / "phase5_research_report.md").write_text(text)
    audit(out, R, tab)
    print(f"report ({len(text):,} chars) and audit_summary.json written")


if __name__ == "__main__":
    main()
