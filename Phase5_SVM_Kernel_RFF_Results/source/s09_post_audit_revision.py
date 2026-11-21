#!/usr/bin/env python3
"""
s09_post_audit_revision.py - POST-AUDIT revision stage (new in remediation).

This stage never edits the original lock (manifests/LOCKED_SELECTION.json) or the original prediction files.
It adds clearly-labelled *revised* artifacts:

  R1  Apply the pre-declared validation-only RFF seed rule (common.select_rff_policy) to the stored 30-run
      Stage C experiment -> manifests/REVISED_SELECTION_post_audit.json   (written BEFORE any test file is opened).
      Also: seed-variability table, and an audit of whether the 0.005 tie rule would have changed any
      original Stage A selection.
  R2  Fit the five pre-declared seeds (42..46) at D* for pca_B and hosvd_B (train only). The rule-selected seed
      additionally gets a train-only calibrator (identical protocol to s06) and aligned prediction CSVs.
      Per-seed validation AND test metrics are stored for transparency (descriptive; never used for selection).
  R3  Stage B (exact RBF) validation-selected candidates are evaluated on validation/test as a supplementary
      post-lock addendum (no calibrated probabilities are produced for them -> left missing, reason recorded).

Honesty note written into every artifact: test labels of the ORIGINAL final models had already been inspected
before this revision existed, so the revised selection cannot be described as "locked before the test".
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DEFAULT_SEEDS, RunRegistry, apply_frozen_scaler, classification_metrics, load_frozen_scaler,
                    select_rff_policy, select_with_tie_rule, transition_metrics, utc_now)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Phase 5 post-audit revision (RFF seed rule, revised finals, Stage B addendum)")
    p.add_argument("--input-dir", default=".")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--force", action="store_true")
    return p.parse_args(argv)


def load_split(in_dir, split, cfg, mean, scale):
    with np.load(in_dir / f"features_{split}_{cfg}.npz") as z:
        return (apply_frozen_scaler(z["F"], mean, scale), z["y"].astype(np.int64),
                z["current_state"].astype(np.int64), z["target_timestamp"])


def pred_frame(ts, y, c, hard, scores, probs, model_id, calib_note):
    df = pd.DataFrame({"target_timestamp": ts, "y": y, "current_state": c, "scope_a_eligible": c >= 0,
                       "hard_pred": hard})
    df["calibrated_argmax"] = np.argmax(probs, axis=1) if probs is not None else np.nan
    for k in range(3):
        df[f"score_{k}"] = scores[:, k]
    for k in range(3):
        df[f"prob_{k}"] = probs[:, k] if probs is not None else np.nan
    df["model_id"] = model_id
    df["calibration_status"] = calib_note
    return df


def main(argv=None):
    args = parse_args(argv)
    in_dir = Path(args.input_dir)
    out = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent.parent
    for s in ("logs", "results", "manifests", "models", "predictions/revised", "predictions/stage_b", "checkpoints/revision"):
        (out / s).mkdir(parents=True, exist_ok=True)
    import os
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s",
                        handlers=[logging.FileHandler(out / "logs" / "s09_post_audit_revision.log", mode="a"),
                                  logging.StreamHandler(sys.stdout)])
    log = logging.getLogger("revision")
    reg = RunRegistry(out / "logs" / "run_registry.jsonl")
    import sklearn
    from sklearn.kernel_approximation import RBFSampler
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    log.info("s09 start %s | sklearn %s | numpy %s", utc_now(), sklearn.__version__, np.__version__)

    # ------------------------------------------------------------------ R1 (validation-only, before test access)
    exp = pd.read_csv(out / "results" / "stage_c_full_experiment.csv")
    assert len(exp) == 30 and set(exp["seed"]) == set(DEFAULT_SEEDS), "Stage C must hold 3 D x 5 seeds x 2 configs"
    pol = select_rff_policy(exp, ("pca_B", "hosvd_B"))
    D_star, seed_star = pol["D_star"], pol["seed_star"]
    orig = json.loads((out / "manifests" / "LOCKED_SELECTION.json").read_text())["selected_models"]
    var_rows = []
    for (cfg, D), g in exp.groupby(["config", "D"]):
        v = g.set_index("seed")["val_scopeB_macro_f1"]
        var_rows.append({"config": cfg, "D": int(D), "n_seeds": len(v), "mean": v.mean(), "std_ddof1": v.std(ddof=1),
                         "min": v.min(), "median": v.median(), "max": v.max(), "range": v.max() - v.min(),
                         **{f"seed_{s}": float(v[s]) for s in sorted(v.index)}})
    pd.DataFrame(var_rows).to_csv(out / "results" / "rff_seed_variability_validation.csv", index=False)
    orig_info = {}
    for key, cfg in (("rff_pca_B", "pca_B"), ("rff_hosvd_B", "hosvd_B")):
        o = orig[key]
        v = exp[(exp["config"] == cfg) & (exp["D"] == o["D"])].set_index("seed")["val_scopeB_macro_f1"].sort_values(ascending=False)
        orig_info[key] = {"D": o["D"], "seed": o["seed"], "val_macro_f1": o["val_scopeB_macro_f1"],
                          "rank_among_5_seeds_best_is_1": int(list(v.index).index(o["seed"]) + 1),
                          "is_best_validation_seed": bool(v.index[0] == o["seed"])}
    revised = {"status": "POST_AUDIT_REVISION", "utc": utc_now(),
               "original_lock_preserved_at": "manifests/LOCKED_SELECTION.json",
               "test_metrics_used_for_selection": False,
               "provenance_caveat": ("The ORIGINAL models' test metrics had already been computed and seen before this revised rule "
                                     "was written. This revision therefore is NOT a pre-test lock; it is a validation-only rule "
                                     "applied after the audit, and its test numbers are reported as a separately-labelled analysis."),
               "rule": pol["rule"], "predeclared_seeds": list(DEFAULT_SEEDS),
               "D_star": D_star, "seed_star": seed_star, "details": pol,
               "original_lock_rff_choices": orig_info,
               "finding": "original lock selected idxmax over single-seed validation rows (best-seed selection): "
                          + json.dumps({k: v['is_best_validation_seed'] for k, v in orig_info.items()}),
               "selected_models": {f"revised_rff_{cfg}": {**v, "model_type": "RFF_LinearSVC_OvR",
                                                          "shared_frequencies_across_pair": True}
                                   for cfg, v in pol["chosen"].items()}}
    (out / "manifests" / "REVISED_SELECTION_post_audit.json").write_text(json.dumps(revised, indent=2, default=float))
    log.info("R1: D*=%d seed*=%d (seed order best->worst %s); original lock picks: %s", D_star, seed_star,
             pol["seed_rank_best_to_worst"], orig_info)

    # tie-rule audit of ORIGINAL Stage A selections (validation only)
    sa = pd.read_csv(out / "results" / "stage_a_all_candidates.csv")
    audit = []
    for cfg, g in sa.groupby("config"):
        recs = [{**r, "cost": r["fit_time_sec"]} for r in g.to_dict("records")]
        a = max(recs, key=lambda r: r["val_scopeB_macro_f1"])
        t = select_with_tie_rule(recs, "val_scopeB_macro_f1", secondary=("val_trans_f1", "val_trans_recall", "val_scopeB_balanced_accuracy"))
        audit.append({"config": cfg, "argmax_C": a["C"], "argmax_val_macro_f1": a["val_scopeB_macro_f1"], "tie_rule_C": t["C"],
                      "tie_rule_val_macro_f1": t["val_scopeB_macro_f1"], "selection_changes": a["C"] != t["C"],
                      "n_within_0.005_of_best": int(sum(a["val_scopeB_macro_f1"] - r["val_scopeB_macro_f1"] <= 0.005 for r in recs))})
    pd.DataFrame(audit).to_csv(out / "results" / "stage_a_tie_rule_audit.csv", index=False)
    log.info("Stage A tie-rule audit: selections that would change = %s", [a["config"] for a in audit if a["selection_changes"]])

    # ------------------------------------------------------------------ R2 revised RFF (train-only fits)
    ck = out / "checkpoints" / "revision"
    sweep_rows = []
    import gc
    for cfg in ("pca_B", "hosvd_B"):
        mean, scale = load_frozen_scaler(out / "models" / f"scaler_{cfg}.npz")
        Xtr, ytr, ctr, _ = load_split(in_dir, "train", cfg, mean, scale)
        Xva, yva, cva, tsv = load_split(in_dir, "validation", cfg, mean, scale)
        ch = pol["chosen"][cfg]
        # --- the test split is opened ONLY now, after REVISED_SELECTION_post_audit.json was written
        Xte, yte, cte, tst = load_split(in_dir, "test", cfg, mean, scale)
        for seed in DEFAULT_SEEDS:
            tag = ck / f"sweep_{cfg}_D{D_star}_s{seed}.npz"
            is_star = seed == seed_star
            have_final = (out / "models" / f"final_revised_rff_{cfg}.pkl").exists() and (out / "predictions" / "revised" / f"final_revised_rff_{cfg}_test.csv").exists()
            if tag.exists() and not args.force and not (is_star and not have_final):
                z = np.load(tag)
                pv, pt, fit_s = z["pv"], z["pt"], float(z["fit_s"])
                log.info("%s seed %d: resumed from checkpoint", cfg, seed)
            else:
                t0 = time.perf_counter()
                smp = RBFSampler(gamma=ch["gamma"], n_components=D_star, random_state=seed)
                Ztr = smp.fit_transform(Xtr)
                clf = LinearSVC(C=ch["C"], loss="squared_hinge", multi_class="ovr", max_iter=5000, random_state=42, dual=False)
                clf.fit(Ztr, ytr)
                fit_s = time.perf_counter() - t0
                Zva, Zte = smp.transform(Xva), smp.transform(Xte)
                pv, pt = clf.predict(Zva), clf.predict(Zte)
                np.savez(tag, pv=pv, pt=pt, fit_s=fit_s)
                log.info("%s seed %d fit (RFF+LinearSVC) %.1fs", cfg, seed, fit_s)
                if is_star:
                    # ---- calibration (train-only chronological tail) + final artifacts, done immediately to bound memory
                    cut = int(len(ytr) * 0.8)
                    base = LinearSVC(C=ch["C"], loss="squared_hinge", multi_class="ovr", max_iter=5000, random_state=42, dual=False)
                    base.fit(Ztr[:cut], ytr[:cut])
                    assert list(base.classes_) == [0, 1, 2] and set(np.unique(ytr[cut:])) == {0, 1, 2}, "class support/order violated"
                    cal = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000).fit(base.decision_function(Ztr[cut:]), ytr[cut:])
                    assert list(cal.classes_) == [0, 1, 2]
                    del base
                    for split, Z, y, c, ts in (("validation", Zva, yva, cva, tsv), ("test", Zte, yte, cte, tst)):
                        sc_, hard = clf.decision_function(Z), clf.predict(Z)
                        P = cal.predict_proba(sc_)
                        assert P.shape == (len(y), 3) and np.allclose(P.sum(1), 1) and np.isfinite(P).all()
                        pred_frame(ts, y, c, hard, sc_, P, f"revised_rff_{cfg}", "train-only 80/20 chronological-tail multinomial LR").to_csv(
                            out / "predictions" / "revised" / f"final_revised_rff_{cfg}_{split}.csv", index=False)
                    with open(out / "models" / f"final_revised_rff_{cfg}.pkl", "wb") as fh:
                        pickle.dump({"sampler": smp, "classifier": clf, "config": cfg, "D": D_star, "seed": seed_star,
                                     "gamma": ch["gamma"], "C": ch["C"], "class_order": [0, 1, 2]}, fh)
                    with open(out / "models" / f"calibrator_revised_rff_{cfg}.pkl", "wb") as fh:
                        pickle.dump(cal, fh)
                    np.savez(out / "models" / f"rff_mapping_revised_{cfg}_D{D_star}_s{seed_star}.npz",
                             random_weights=smp.random_weights_, random_offset=smp.random_offset_, gamma=ch["gamma"])
                    log.info("%s revised final (D=%d, seed=%d) saved with train-only calibrator", cfg, D_star, seed_star)
                    reg.log(stage="revision", config=cfg, status="done", D=D_star, seed=seed_star)
                del Ztr, Zva, Zte, clf, smp
                gc.collect()
            for split, y, c, p in (("validation", yva, cva, pv), ("test", yte, cte, pt)):
                m, tm = classification_metrics(y, p), transition_metrics(y, p, c)
                sweep_rows.append({"config": cfg, "D": D_star, "seed": seed, "split": split, "selected_by_rule": is_star,
                                   "fit_time_s": fit_s, "scopeB_accuracy": m["accuracy"], "scopeB_macro_f1": m["macro_f1"],
                                   "scopeA_trans_f1": tm["trans_f1"], "scopeA_trans_recall": tm["trans_recall"],
                                   "scopeA_high_recall": tm["high_recall"], "scopeA_high_f1": tm["high_f1"],
                                   "use": "descriptive seed-variability only; not used for selection"})
        del Xtr, Xva, Xte
        gc.collect()
    pd.DataFrame(sweep_rows).to_csv(out / "results" / "rff_seed_sweep_descriptive.csv", index=False)

    # ------------------------------------------------------------------ R3 Stage B addendum
    sel_path = out / "manifests" / "STAGE_B_SELECTION_validation_only.json"
    if sel_path.exists():
        sel = json.loads(sel_path.read_text())["per_config"]
        for cfg in ("pca_B", "hosvd_B"):
            s = sel.get(cfg)
            if not s:
                log.warning("Stage B: no validation-selected candidate for %s", cfg)
                continue
            mean, scale = load_frozen_scaler(out / "models" / f"scaler_{cfg}.npz")
            with open(out / "checkpoints" / "stage_b" / f"{s['run_id']}.pkl", "rb") as fh:
                clf = pickle.load(fh)
            for split in ("validation", "test"):
                X, y, c, ts = load_split(in_dir, split, cfg, mean, scale)
                t0 = time.perf_counter()
                hard, sc = clf.predict(X), clf.decision_function(X)
                pt_s = time.perf_counter() - t0
                pred_frame(ts, y, c, hard, sc, None, f"stageB_{cfg}", "MISSING: no probability calibrator was fitted for exact SVC "
                           "(needs chronological-fold refits; not run)").to_csv(
                    out / "predictions" / "stage_b" / f"final_stageB_{cfg}_{split}.csv", index=False)
                log.info("Stage B %s %s predicted in %.2fs (run %s)", cfg, split, pt_s, s["run_id"])
            with open(out / "models" / f"final_stageB_{cfg}.pkl", "wb") as fh:
                pickle.dump({"classifier": clf, "run_id": s["run_id"], "class_order_via_classes_": [0, 1, 2]}, fh)
        full_csv = out / "results" / "stage_b_full_train_exact_pilot.csv"
        if full_csv.exists():
            fdf = pd.read_csv(full_csv)
            for _, r in fdf[fdf["status"] == "done"].iterrows():
                cfg = r["config"]
                mean, scale = load_frozen_scaler(out / "models" / f"scaler_{cfg}.npz")
                with open(out / "checkpoints" / "stage_b" / f"{r['run_id']}.pkl", "rb") as fh:
                    clf = pickle.load(fh)
                for split in ("validation", "test"):
                    X, y, c, ts = load_split(in_dir, split, cfg, mean, scale)
                    hard, sc = clf.predict(X), clf.decision_function(X)
                    pred_frame(ts, y, c, hard, sc, None, f"stageB_full_{cfg}", "MISSING: no probability calibrator was fitted for exact SVC").to_csv(
                        out / "predictions" / "stage_b" / f"final_stageB_full_{cfg}_{split}.csv", index=False)
                with open(out / "models" / f"final_stageB_full_{cfg}.pkl", "wb") as fh:
                    pickle.dump({"classifier": clf, "run_id": r["run_id"]}, fh)
                log.info("Stage B FULL-train exact %s evaluated (hyper-parameters from fit_n=8000 validation winner)", cfg)
    else:
        log.warning("Stage B selection manifest absent -> Stage B addendum skipped (run stage_b first)")
    reg.log(stage="revision", status="stage_complete")
    log.info("s09 complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
