#!/usr/bin/env python3
"""
s10_tables_and_bootstrap.py - authoritative aggregate tables + corrected paired block bootstrap (new in remediation).

Single source of truth: every number produced here is recomputed from the saved prediction CSVs
(predictions/**.csv) and the supplied Phase 3 Markov CSVs by an implementation that is independent of the
one inside s06 (common.py, cross-checked against sklearn in the tests).  Outputs:

  results/final_metrics_from_predictions.csv      one row per (model, split)
  results/reconciliation_vs_final_evaluation.csv  max |difference| vs the original s06 table
  results/bootstrap_ci.json / bootstrap_ci_table.csv   timestamp-based calendar blocks (24 h primary, 168 h sensitivity)

Nothing here fits a model or selects anything.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import classification_metrics, paired_block_bootstrap, prob_metrics, transition_metrics, utc_now


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Aggregate tables + paired block bootstrap from prediction CSVs")
    p.add_argument("--input-dir", default=".")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    return p.parse_args(argv)


FAMILIES = [("original_lock", "predictions", "final_{}_{}.csv", ["linear_pca_B", "linear_hosvd_B", "linear_best_overall", "rff_pca_B", "rff_hosvd_B"]),
            ("post_audit_revision", "predictions/revised", "final_{}_{}.csv", ["revised_rff_pca_B", "revised_rff_hosvd_B"]),
            ("stage_b_addendum", "predictions/stage_b", "final_{}_{}.csv", ["stageB_pca_B", "stageB_hosvd_B", "stageB_full_pca_B", "stageB_full_hosvd_B"])]


def load_predictions(out: Path):
    preds = {}
    for fam, d, pat, ids in FAMILIES:
        for mid in ids:
            for split in ("validation", "test"):
                p = out / d / pat.format(mid, split)
                if p.exists():
                    df = pd.read_csv(p, parse_dates=["target_timestamp"])
                    df["family"] = fam
                    preds[(mid, split)] = df
    return preds


def check_alignment(preds, markov):
    """Hard checks: identical rows (timestamp, y, current_state, eligibility) across all models and Phase 3 CSVs."""
    for split in ("validation", "test"):
        m = markov[split]
        for (mid, s), df in preds.items():
            if s != split:
                continue
            assert len(df) == len(m), f"{mid} {split}: {len(df)} rows vs Markov {len(m)}"
            assert np.array_equal(df["target_timestamp"].values, m["target_timestamp"].values), f"{mid} {split}: timestamps differ"
            assert np.array_equal(df["y"].values, m["y"].values), f"{mid} {split}: y differs"
            assert np.array_equal(df["current_state"].values, m["current_state"].values), f"{mid} {split}: current_state differs"
            assert np.array_equal(df["scope_a_eligible"].values.astype(bool), m["scope_a_eligible"].values.astype(bool)), f"{mid} {split}: eligibility differs"
        assert m["target_timestamp"].is_unique, f"duplicate timestamps in Markov {split}"


def metric_row(model, family, split, y, c, hard, P=None, cal_arg=None):
    a = c >= 0
    mb, ma = classification_metrics(y, hard), classification_metrics(y[a], hard[a])
    tm = transition_metrics(y, hard, c)
    r = {"model_id": model, "family": family, "split": split, "scopeB_n": mb["n"], "scopeA_n": ma["n"],
         "scopeB_accuracy": mb["accuracy"], "scopeB_macro_f1": mb["macro_f1"], "scopeB_weighted_f1": mb["weighted_f1"],
         "scopeB_balanced_accuracy": mb["balanced_accuracy"],
         "scopeA_accuracy": ma["accuracy"], "scopeA_macro_f1": ma["macro_f1"], "scopeA_balanced_accuracy": ma["balanced_accuracy"]}
    for k in ("true_transitions", "pred_transitions", "trans_tp", "trans_fp", "trans_fn", "trans_tn", "trans_precision", "trans_recall",
              "trans_f1", "trans_dest_accuracy", "trans_stable_accuracy", "true_high_entries", "pred_high_entries", "high_tp", "high_fp",
              "high_fn", "high_precision", "high_recall", "high_f1"):
        r[k] = tm[k]
    if P is not None and np.isfinite(P).all():
        pb, pa = prob_metrics(y, P), prob_metrics(y[a], P[a])
        r.update({"scopeB_brier": pb["brier"], "scopeB_log_loss": pb["log_loss"], "scopeA_brier": pa["brier"], "scopeA_log_loss": pa["log_loss"]})
        if cal_arg is not None:
            r["scopeB_calibrated_argmax_accuracy"] = float((cal_arg == y).mean())
            r["calibrated_argmax_differs_from_hard_frac"] = float((cal_arg != hard).mean())
    else:
        r.update({"scopeB_brier": np.nan, "scopeB_log_loss": np.nan, "scopeA_brier": np.nan, "scopeA_log_loss": np.nan})
    return r


def main(argv=None):
    args = parse_args(argv)
    in_dir = Path(args.input_dir)
    out = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent.parent
    (out / "results").mkdir(exist_ok=True, parents=True)
    (out / "logs").mkdir(exist_ok=True, parents=True)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s",
                        handlers=[logging.FileHandler(out / "logs" / "s10_tables_and_bootstrap.log", mode="w"), logging.StreamHandler(sys.stdout)])
    log = logging.getLogger("s10")
    markov = {s: pd.read_csv(in_dir / f"markov_predictions_{s}.csv", parse_dates=["target_timestamp"]) for s in ("validation", "test")}
    preds = load_predictions(out)
    log.info("loaded %d prediction files: %s", len(preds), sorted({k[0] for k in preds}))
    check_alignment(preds, markov)
    log.info("alignment with Phase 3 Markov CSVs verified row-for-row (timestamp, y, current_state, eligibility)")

    rows = []
    for (mid, split), df in sorted(preds.items()):
        y, c, h = df["y"].values.astype(int), df["current_state"].values.astype(int), df["hard_pred"].values.astype(int)
        P = df[["prob_0", "prob_1", "prob_2"]].values.astype(float)
        rows.append(metric_row(mid, df["family"].iloc[0], split, y, c, h, P, df["calibrated_argmax"].values if np.isfinite(P).all() else None))
    for split, m in markov.items():
        y, c = m["y"].values.astype(int), m["current_state"].values.astype(int)
        for name, col, pre in (("phase3_majority", "pred_majority", "prob_majority"), ("phase3_persistence", "pred_persistence_scope_b", None),
                               ("phase3_markov", "pred_markov_scope_b", "prob_markov")):
            P = m[[f"{pre}_{k}" for k in range(3)]].values.astype(float) if pre else None
            rows.append(metric_row(name, "phase3_baseline", split, y, c, m[col].values.astype(int), P))
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "results" / "final_metrics_from_predictions.csv", index=False)
    log.info("wrote final_metrics_from_predictions.csv (%d rows)", len(tab))

    # Phase 3 reference counts recomputed from CSV (audit findings if they differ)
    ref = {}
    for split, m in markov.items():
        a = m[m["scope_a_eligible"].astype(bool)]
        ref[split] = {"scopeA_n": int(len(a)), "scopeB_n": int(len(m)), "actual_transitions": int((a["y"] != a["current_state"]).sum()),
                      "entries_into_high": int(((a["current_state"] != 2) & (a["y"] == 2)).sum())}
    t = tab[(tab.model_id == "phase3_markov")]
    ref["markov_test_scopeA_accuracy"] = float(t[t.split == "test"]["scopeA_accuracy"].iloc[0])
    ref["markov_test_scopeB_accuracy"] = float(t[t.split == "test"]["scopeB_accuracy"].iloc[0])
    ref["markov_test_hard_detected_changes"] = int(t[t.split == "test"]["pred_transitions"].iloc[0])
    ref["markov_test_scopeA_brier"] = float(t[t.split == "test"]["scopeA_brier"].iloc[0])
    ref["markov_test_scopeA_log_loss"] = float(t[t.split == "test"]["scopeA_log_loss"].iloc[0])
    (out / "results" / "phase3_reference_recomputed.json").write_text(json.dumps(ref, indent=2))
    log.info("Phase 3 reference recomputed: %s", ref)

    # reconciliation with the original s06 table
    orig = pd.read_csv(out / "results" / "final_evaluation_scopeAB.csv")
    cols = ["scopeB_accuracy", "scopeB_macro_f1", "scopeB_weighted_f1", "scopeB_balanced_accuracy", "scopeB_brier", "scopeB_log_loss",
            "scopeA_accuracy", "scopeA_macro_f1", "scopeA_brier", "scopeA_log_loss", "trans_precision", "trans_recall", "trans_f1",
            "trans_dest_accuracy", "high_precision", "high_recall", "high_f1", "trans_tp", "trans_fp", "trans_fn", "high_tp", "high_fp", "high_fn"]
    rec = []
    for _, o in orig.iterrows():
        n = tab[(tab.model_id == o["model_id"]) & (tab.split == o["split"])].iloc[0]
        d = {c: abs(float(o[c]) - float(n[c])) for c in cols}
        rec.append({"model_id": o["model_id"], "split": o["split"], "max_abs_diff": max(d.values()), "worst_column": max(d, key=d.get)})
    rec = pd.DataFrame(rec)
    rec.to_csv(out / "results" / "reconciliation_vs_final_evaluation.csv", index=False)
    log.info("reconciliation vs original final_evaluation_scopeAB.csv: max abs diff = %.3e", rec["max_abs_diff"].max())

    # ------------------------------------------------------------------ paired block bootstrap on TEST
    def T(m):
        return preds[(m, "test")]
    pairs_spec = [("linear_pca_B", "linear_hosvd_B", "Linear SVM: PCA-B minus HoSVD-B (original lock)"),
                  ("revised_rff_pca_B", "revised_rff_hosvd_B", "RFF SVM: PCA-B minus HoSVD-B (revised common seed)"),
                  ("linear_pca_B", "revised_rff_pca_B", "PCA-B: linear minus RFF"),
                  ("linear_hosvd_B", "revised_rff_hosvd_B", "HoSVD-B: linear minus RFF"),
                  ("stageB_pca_B", "stageB_hosvd_B", "Exact RBF SVC (fit_n subset): PCA-B minus HoSVD-B"),
                  ("revised_rff_pca_B", "stageB_pca_B", "PCA-B: RFF (full train) minus exact RBF (subset)"),
                  ("stageB_full_pca_B", "stageB_full_hosvd_B", "Exact RBF SVC (FULL train): PCA-B minus HoSVD-B"),
                  ("stageB_full_pca_B", "revised_rff_pca_B", "PCA-B: exact RBF (full train) minus RFF (full train)"),
                  ("stageB_full_pca_B", "linear_pca_B", "PCA-B: exact RBF (full train) minus linear")]
    pairs_spec = [p for p in pairs_spec if (p[0], "test") in preds and (p[1], "test") in preds]
    ids_needed = sorted({m for p in pairs_spec for m in p[:2]})
    base = T(ids_needed[0])
    P = {m: T(m)["hard_pred"].values.astype(int) for m in ids_needed}
    for m in ids_needed:
        assert np.array_equal(T(m)["target_timestamp"].values, base["target_timestamp"].values)
    result = {"utc": utc_now(), "note": ("Test-set sampling uncertainty only (blocks of real time, resampled with replacement, same blocks for both models). "
                                         "Does NOT include training-set variability or RFF seed variability. Intervals are percentile 95%; "
                                         "'includes zero' is reported literally and no significance test is implied."),
              "superseded_original_artifact": {"file": "superseded_pre_audit/results/bootstrap_ci.json", "block_definition": "24 consecutive ROW-INDEX blocks (239 blocks)",
                                                "transition_f1_diff_ci95": (json.loads((out / "superseded_pre_audit/results/bootstrap_ci.json").read_text())["transition_f1"]["ci_95"]
                                                                       if (out / "superseded_pre_audit/results/bootstrap_ci.json").exists() else None)},
              "runs": {}}
    flat = []
    for hours in (24, 168):
        res = paired_block_bootstrap(base["y"].values.astype(int), base["current_state"].values.astype(int), P, base["target_timestamp"].values,
                                     [(a, b) for a, b, _ in pairs_spec], block_hours=hours, n_boot=args.n_bootstrap, seed=args.seed)
        result["runs"][f"block_{hours}h"] = res
        for (a, b, desc) in pairs_spec:
            for metric, v in res["pairs"][f"{a} - {b}"].items():
                flat.append({"block_hours": hours, "n_blocks": res["n_blocks"], "comparison": desc, "model_a": a, "model_b": b, "metric": metric,
                             "observed_a": v["observed_a"], "observed_b": v["observed_b"], "observed_diff_a_minus_b": v["observed_diff_a_minus_b"],
                             "ci95_low": v["ci95"][0], "ci95_high": v["ci95"][1], "se": v["se"], "ci_includes_zero": v["ci_includes_zero"]})
    (out / "results" / "bootstrap_ci.json").write_text(json.dumps(result, indent=2))
    pd.DataFrame(flat).to_csv(out / "results" / "bootstrap_ci_table.csv", index=False)
    for r in flat:
        if r["block_hours"] == 24 and r["metric"] in ("macro_f1", "trans_f1"):
            log.info("[24h] %-62s %-9s diff=%+.4f CI95=[%+.4f, %+.4f] includes0=%s", r["comparison"], r["metric"],
                     r["observed_diff_a_minus_b"], r["ci95_low"], r["ci95_high"], r["ci_includes_zero"])
    log.info("s10 complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
