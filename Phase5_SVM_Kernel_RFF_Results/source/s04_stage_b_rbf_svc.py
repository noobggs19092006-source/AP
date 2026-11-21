#!/usr/bin/env python3
"""
s04_stage_b_rbf_svc.py - Stage B: exact RBF kernel SVC (native one-vs-one libsvm).   [REWRITTEN in remediation]

What changed relative to the audited version (kept in superseded_pre_audit/)
---------------------------------------------------------------------------
* Scaler loading goes through common.load_frozen_scaler (accepts the real Stage A keys ``mean``/``scale``;
  raises if the scaler is missing; there is NO fallback fit on validation/test or on anything else).
* Portable CLI (no absolute paths).  Resumable: every grid cell writes its own JSON status file; completed
  cells are skipped unless --force.
* Training subsets are deterministic systematic temporal-block samples spread over the whole training period
  (the old "first N rows" subset covered only the earliest part of training).  Class mix, time coverage and
  shift from the full training distribution are written to results/stage_b_subsets.csv.
* Each fit runs in a forked child process with an ENFORCED wall-time limit (--rbf-max-seconds); a timeout is
  recorded as ``skipped_timeout``.  RAM is guarded by an up-front estimate (--max-ram-gb); peak RSS is
  MEASURED (getrusage in the child) but the RAM limit itself is not an OS-enforced cap.
* The old "OvR vs OvO" ablation only toggled decision_function_shape, which cannot change SVC.predict.  It is
  replaced by a genuine comparison: native SVC (OvO) vs OneVsRestClassifier(SVC) trained on the same subset,
  same scaler, same gamma/C; the old toggle is also run to demonstrate that it is a no-op.
* Selection of the Stage B candidate uses VALIDATION only (Scope B macro-F1 with the 0.005 tie rule).
  No test data are touched here.

Grid (spec section 7): representations pca_B / hosvd_B (d=192), gamma = a/d with a in {0.1,1,10},
C in {0.1,1,10}, fit_n in {4000, 8000}  ->  2 x 2 x 9 = 36 candidate fits.
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import pickle
import queue
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CLASS_NAMES, RunRegistry, apply_frozen_scaler, classification_metrics, current_rss_mb,
                    load_frozen_scaler, select_with_tie_rule, temporal_block_subset, transition_metrics, utc_now)

D_FEAT = 192
GAMMA_SCALES = (0.1, 1.0, 10.0)
C_VALUES = (0.1, 1.0, 10.0)
FIT_NS = (4000, 8000)
CONFIGS = ("pca_B", "hosvd_B")
BLOCK_LEN = 100


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Stage B: exact RBF SVC (bounded, resumable)")
    p.add_argument("--input-dir", default=".", help="Directory with features_*.npz (Phase 4)")
    p.add_argument("--output-dir", default=None, help="Package root (default: parent of source/)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=1, help="BLAS threads (libsvm itself is single-threaded)")
    p.add_argument("--max-ram-gb", type=float, default=2.5, help="Skip a fit whose estimated peak exceeds this")
    p.add_argument("--rbf-max-seconds", type=float, default=600.0, help="Enforced wall-time limit per fit")
    p.add_argument("--stage-b-budget-seconds", type=float, default=3600.0, help="Total wall budget for the grid")
    p.add_argument("--cache-mb", type=int, default=500, help="libsvm kernel cache (MB)")
    p.add_argument("--fit-ns", type=int, nargs="+", default=list(FIT_NS))
    p.add_argument("--full-exact", action="store_true",
                   help="OPTIONAL: pilot a full 24,108-row exact fit at the fit_n=8000 validation-winner "
                        "hyper-parameters (validation only; bounded by --rbf-max-seconds)")
    p.add_argument("--force", action="store_true", help="Recompute cells even if a status file exists")
    return p.parse_args(argv)


# -------------------------------------------------------------------------- worker (child process)
def _fit_worker(q, X, y, Xv, params, kind, ckpt_path, parent_rss_mb):
    """Fit in a child so wall time can be enforced. Returns arrays/statistics through the queue."""
    import resource
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.svm import SVC
    try:
        base = SVC(kernel="rbf", gamma=params["gamma"], C=params["C"], cache_size=params["cache_mb"],
                   random_state=params["seed"], decision_function_shape=params.get("dfs", "ovr"))
        clf = OneVsRestClassifier(base, n_jobs=1) if kind == "ovr_trained" else base
        t0 = time.perf_counter()
        clf.fit(X, y)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = clf.predict(Xv)
        scores = clf.decision_function(Xv)
        pred_s = time.perf_counter() - t0
        if kind == "ovr_trained":
            n_sv, sv_pc = int(sum(e.n_support_.sum() for e in clf.estimators_)), None
        else:
            n_sv, sv_pc = int(clf.n_support_.sum()), [int(v) for v in clf.n_support_]
        blob = pickle.dumps(clf, protocol=pickle.HIGHEST_PROTOCOL)
        if ckpt_path:
            Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
            Path(ckpt_path).write_bytes(blob)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_mb = peak / 1024 ** 2 if sys.platform == "darwin" else peak / 1024
        q.put({"ok": True, "pred": pred.astype(np.int8), "scores": scores.astype(np.float32), "fit_s": fit_s,
               "predict_val_s": pred_s, "n_sv": n_sv, "sv_per_class": sv_pc, "model_bytes": len(blob),
               "child_peak_rss_mb": peak_mb, "parent_rss_at_fork_mb": parent_rss_mb})
    except BaseException as e:  # noqa: BLE001
        q.put({"ok": False, "error": f"{type(e).__name__}: {e}"})


def run_bounded(X, y, Xv, params, kind, ckpt_path, max_seconds):
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    proc = ctx.Process(target=_fit_worker, args=(q, X, y, Xv, params, kind, ckpt_path, current_rss_mb()))
    t0 = time.perf_counter()
    proc.start()
    try:
        res = q.get(timeout=max_seconds)
    except queue.Empty:
        res = None
    if res is None:
        proc.terminate()
        proc.join()
        return {"ok": False, "timeout": True, "wall_s": time.perf_counter() - t0}
    proc.join()
    res["wall_s"] = time.perf_counter() - t0
    return res


def est_peak_mb(n, d, cache_mb):
    """Conservative upper estimate: feature copy + SV copy + kernel cache (no dense Gram matrix is built)."""
    return n * d * 8 * 2 / 1024 ** 2 + cache_mb + 150


def class_dist(y):
    cnt = np.bincount(y, minlength=3)
    return cnt, cnt / cnt.sum()


# -------------------------------------------------------------------------- main
def main(argv=None):
    args = parse_args(argv)
    in_dir = Path(args.input_dir)
    out = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent.parent
    for sub in ("logs", "results", "figures", "checkpoints/stage_b", "models", "manifests"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(out / "logs" / "s04_stage_b.log", mode="a"),
                                  logging.StreamHandler(sys.stdout)])
    log = logging.getLogger("stage_b")
    reg = RunRegistry(out / "logs" / "run_registry.jsonl")
    import sklearn
    log.info("=" * 70)
    log.info("Stage B start %s | sklearn %s | numpy %s | threads=%d | rbf-max-seconds=%.0f | max-ram-gb=%.1f",
             utc_now(), sklearn.__version__, np.__version__, args.threads, args.rbf_max_seconds, args.max_ram_gb)

    # ---- load train/validation features and FROZEN scalers (no fitting anywhere in this stage)
    D = {}
    for cfg in CONFIGS:
        mean, scale = load_frozen_scaler(out / "models" / f"scaler_{cfg}.npz")
        with np.load(in_dir / f"features_train_{cfg}.npz") as z:
            Ftr, ytr, tstr = z["F"], z["y"].astype(np.int64), z["target_timestamp"]
        with np.load(in_dir / f"features_validation_{cfg}.npz") as z:
            Fva, yva, csva = z["F"], z["y"].astype(np.int64), z["current_state"]
        D[cfg] = dict(Xtr=apply_frozen_scaler(Ftr, mean, scale), ytr=ytr, ts=tstr,
                      Xva=apply_frozen_scaler(Fva, mean, scale), yva=yva, cva=csva.astype(np.int64))
        assert D[cfg]["Xtr"].shape[1] == D_FEAT and np.isfinite(D[cfg]["Xtr"]).all()
        log.info("%s: train %s, val %s scaled with the FROZEN Stage-A scaler (fitted on train only)", cfg,
                 D[cfg]["Xtr"].shape, D[cfg]["Xva"].shape)
    assert np.array_equal(D["pca_B"]["yva"], D["hosvd_B"]["yva"]) and np.array_equal(D["pca_B"]["cva"], D["hosvd_B"]["cva"])
    assert np.array_equal(D["pca_B"]["ytr"], D["hosvd_B"]["ytr"])
    yva, cva = D["pca_B"]["yva"], D["pca_B"]["cva"]
    N_tr = len(D["pca_B"]["ytr"])

    # ---- subsets (identical row indices for both representations) + coverage report
    subsets, rows = {}, []
    full_cnt, full_share = class_dist(D["pca_B"]["ytr"])
    ts = D["pca_B"]["ts"]
    for n in args.fit_ns:
        idx = temporal_block_subset(N_tr, n, BLOCK_LEN)
        subsets[n] = idx
        cnt, share = class_dist(D["pca_B"]["ytr"][idx])
        months = np.unique(ts[idx].astype("datetime64[M]"))
        rows.append({"fit_n": n, "scheme": f"systematic temporal blocks: {BLOCK_LEN} consecutive rows each, evenly spaced over the training period",
                     "n_blocks": n // BLOCK_LEN, "first_timestamp": str(ts[idx].min()), "last_timestamp": str(ts[idx].max()),
                     "distinct_calendar_months_covered": int(len(months)),
                     "full_train_calendar_months": int(len(np.unique(ts.astype("datetime64[M]")))),
                     **{f"n_{CLASS_NAMES[k]}": int(cnt[k]) for k in range(3)},
                     **{f"share_{CLASS_NAMES[k]}": float(share[k]) for k in range(3)},
                     **{f"fulltrain_share_{CLASS_NAMES[k]}": float(full_share[k]) for k in range(3)},
                     "total_variation_distance_vs_full_train": float(0.5 * np.abs(share - full_share).sum())})
        log.info("subset fit_n=%d: class counts %s (full-train shares %s), months %d/%d, TV distance %.4f", n, cnt.tolist(),
                 np.round(full_share, 4).tolist(), rows[-1]["distinct_calendar_months_covered"],
                 rows[-1]["full_train_calendar_months"], rows[-1]["total_variation_distance_vs_full_train"])
    pd.DataFrame(rows).to_csv(out / "results" / "stage_b_subsets.csv", index=False)

    status_dir = out / "checkpoints" / "stage_b"
    t_start = time.perf_counter()
    all_rows = []

    def evaluate(res):
        pa = res["pred"].astype(np.int64)
        return (classification_metrics(yva, pa), classification_metrics(yva[cva >= 0], pa[cva >= 0]),
                transition_metrics(yva, pa, cva))

    def record(run_id, base, res):
        row = dict(base)
        if res.get("ok"):
            mb, ma, tm = evaluate(res)
            row.update({"status": "done", "fit_time_s": res["fit_s"], "predict_val_time_s": res["predict_val_s"],
                        "wall_time_s": res["wall_s"], "n_support_vectors": res["n_sv"],
                        "sv_per_class": json.dumps(res["sv_per_class"]), "model_bytes": res["model_bytes"],
                        "child_peak_rss_mb": res["child_peak_rss_mb"],
                        "peak_rss_increase_mb_est": res["child_peak_rss_mb"] - res["parent_rss_at_fork_mb"],
                        "val_scopeB_n": mb["n"], "val_scopeA_n": ma["n"]})
            for k in ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy"):
                row[f"val_scopeB_{k}"], row[f"val_scopeA_{k}"] = mb[k], ma[k]
            for k in range(3):
                row[f"val_scopeB_{CLASS_NAMES[k]}_f1"] = mb[f"f1_{k}"]
            for k in ("trans_precision", "trans_recall", "trans_f1", "trans_dest_accuracy", "high_precision",
                      "high_recall", "high_f1", "trans_tp", "trans_fp", "trans_fn", "pred_transitions"):
                row[f"val_{k}"] = tm[k]
            np.savez_compressed(status_dir / f"{run_id}_valpred.npz", pred=res["pred"], scores=res["scores"])
        else:
            row.update({"status": "skipped_timeout" if res.get("timeout") else "failed_error",
                        "skip_reason": (f"exceeded --rbf-max-seconds={args.rbf_max_seconds:.0f}s (terminated after "
                                        f"{res.get('wall_s', 0):.0f}s)" if res.get("timeout") else res.get("error"))})
        (status_dir / f"{run_id}.json").write_text(json.dumps(row, default=float, indent=1))
        return row

    # ---- grid
    for fit_n in args.fit_ns:
        for cfg in CONFIGS:
            for a in GAMMA_SCALES:
                for C in C_VALUES:
                    run_id = f"B_{cfg}_n{fit_n}_a{a}_C{C}"
                    base = {"run_id": run_id, "model_family": "exact_RBF_SVC_OvO_libsvm", "config": cfg, "d": D_FEAT,
                            "fit_n": fit_n, "gamma_scale": a, "gamma": a / D_FEAT, "C": C, "seed": args.seed,
                            "cost": fit_n, "utc": utc_now()}
                    jf = status_dir / f"{run_id}.json"
                    if jf.exists() and not args.force:
                        all_rows.append(json.loads(jf.read_text()))
                        log.info("%s: already %s (resume; not recomputed)", run_id, all_rows[-1]["status"])
                        continue
                    skip = None
                    if time.perf_counter() - t_start > args.stage_b_budget_seconds:
                        skip = ("skipped_budget", f"stage budget {args.stage_b_budget_seconds:.0f}s exhausted")
                    elif est_peak_mb(fit_n, D_FEAT, args.cache_mb) > args.max_ram_gb * 1024:
                        skip = ("skipped_ram_guard", f"estimated peak {est_peak_mb(fit_n, D_FEAT, args.cache_mb):.0f} MB > {args.max_ram_gb:.1f} GB guard")
                    if skip:
                        row = {**base, "status": skip[0], "skip_reason": skip[1]}
                        jf.write_text(json.dumps(row, default=float))
                        all_rows.append(row)
                        reg.log(stage="B", run_id=run_id, status=row["status"])
                        log.warning("%s: %s - %s", run_id, *skip)
                        continue
                    idx = subsets[fit_n]
                    res = run_bounded(D[cfg]["Xtr"][idx], D[cfg]["ytr"][idx], D[cfg]["Xva"],
                                      {"gamma": a / D_FEAT, "C": C, "seed": args.seed, "cache_mb": args.cache_mb},
                                      "ovo_native", status_dir / f"{run_id}.pkl", args.rbf_max_seconds)
                    row = record(run_id, base, res)
                    all_rows.append(row)
                    reg.log(stage="B", run_id=run_id, status=row["status"], fit_time_s=row.get("fit_time_s"))
                    if row["status"] == "done":
                        log.info("%s: DONE fit=%.2fs nSV=%d | val ScopeB macroF1=%.4f acc=%.4f | ScopeA transF1=%.4f | child peak RSS=%.0f MB",
                                 run_id, row["fit_time_s"], row["n_support_vectors"], row["val_scopeB_macro_f1"],
                                 row["val_scopeB_accuracy"], row["val_trans_f1"], row["child_peak_rss_mb"])
                    else:
                        log.warning("%s: %s - %s", run_id, row["status"], row["skip_reason"])

    df = pd.DataFrame(all_rows)
    df.to_csv(out / "results" / "stage_b_all_candidates.csv", index=False)
    n_done = int((df["status"] == "done").sum())
    log.info("Grid complete: %d/%d done; statuses: %s", n_done, len(df), df["status"].value_counts().to_dict())

    # ---- validation-only selection
    sel = {"selection_rule": "validation Scope B macro-F1; ties within 0.005 -> Scope A transition F1, recall, "
                             "balanced accuracy, then smaller fit_n (cost). No test data used.",
           "utc": utc_now(), "per_config": {}, "per_config_and_fit_n": {}}
    done = df[df["status"] == "done"]
    keys = ("run_id", "config", "fit_n", "gamma_scale", "gamma", "C", "val_scopeB_macro_f1", "val_scopeB_accuracy",
            "val_trans_f1", "n_support_vectors", "fit_time_s")
    for cfg in CONFIGS:
        pools = [(cfg, done[done["config"] == cfg], "per_config")] + [
            (f"{cfg}|fit_n={n}", done[(done["config"] == cfg) & (done["fit_n"] == n)], "per_config_and_fit_n") for n in args.fit_ns]
        for name, sub, bucket in pools:
            if sub.empty:
                sel[bucket][name] = None
                continue
            r = select_with_tie_rule(sub.to_dict("records"), primary="val_scopeB_macro_f1",
                                     secondary=("val_trans_f1", "val_trans_recall", "val_scopeB_balanced_accuracy"), cost_key="cost")
            sel[bucket][name] = {k: r[k] for k in keys}
    (out / "manifests" / "STAGE_B_SELECTION_validation_only.json").write_text(json.dumps(sel, indent=2, default=float))

    # ---- genuine OvR-vs-OvO ablation at the smallest fit_n, winner (gamma, C) per representation
    abl = []
    n_abl = min(args.fit_ns)
    for cfg in CONFIGS:
        s = sel["per_config_and_fit_n"].get(f"{cfg}|fit_n={n_abl}")
        if not s:
            log.warning("ablation skipped for %s: no completed fit_n=%d cell", cfg, n_abl)
            continue
        idx = subsets[n_abl]
        params = {"gamma": s["gamma"], "C": s["C"], "seed": args.seed, "cache_mb": args.cache_mb}
        variants = [("OvO_native_libsvm", "ovo_native", "ovr"),
                    ("OvO_native_with_dfs_ovo_toggle (old 'ablation')", "ovo_native", "ovo"),
                    ("OvR_trained_OneVsRestClassifier", "ovr_trained", "ovr")]
        preds = {}
        for name, kind, dfs in variants:
            res = run_bounded(D[cfg]["Xtr"][idx], D[cfg]["ytr"][idx], D[cfg]["Xva"], {**params, "dfs": dfs}, kind, None,
                              args.rbf_max_seconds)
            if not res.get("ok"):
                abl.append({"config": cfg, "variant": name, "status": "skipped_timeout" if res.get("timeout") else "failed_error",
                            "skip_reason": res.get("error", "timeout")})
                continue
            mb, ma, tm = evaluate(res)
            preds[name] = res["pred"]
            abl.append({"config": cfg, "variant": name, "status": "done", "fit_n": n_abl, "gamma": params["gamma"], "C": params["C"],
                        "val_scopeB_macro_f1": mb["macro_f1"], "val_scopeB_accuracy": mb["accuracy"],
                        "val_scopeB_balanced_accuracy": mb["balanced_accuracy"], "val_trans_f1": tm["trans_f1"],
                        "val_trans_recall": tm["trans_recall"], "val_high_recall": tm["high_recall"],
                        "fit_time_s": res["fit_s"], "n_support_vectors_total": res["n_sv"], "model_bytes": res["model_bytes"],
                        "child_peak_rss_mb": res["child_peak_rss_mb"],
                        "note": ("SVC.predict is always OvO voting; decision_function_shape only reshapes decision_function" if "toggle" in name
                                 else ("3 independent binary SVCs, each trained on all fit_n rows" if "OvR" in name
                                       else "3 pairwise SVCs, each trained on its two classes only"))})
            log.info("ablation %s | %s: macroF1=%.4f acc=%.4f nSV=%d fit=%.2fs", cfg, name, mb["macro_f1"], mb["accuracy"], res["n_sv"], res["fit_s"])
        k1, k2 = "OvO_native_libsvm", "OvO_native_with_dfs_ovo_toggle (old 'ablation')"
        if k1 in preds and k2 in preds:
            same = bool(np.array_equal(preds[k1], preds[k2]))
            abl.append({"config": cfg, "variant": "CHECK: old dfs-toggle predictions identical to native OvO", "status": "done",
                        "note": f"identical={same}"})
            log.info("ablation %s: old decision_function_shape toggle gives identical predictions: %s", cfg, same)
    pd.DataFrame(abl).to_csv(out / "results" / "stage_b_ovr_ovo_ablation.csv", index=False)

    # ---- OPTIONAL full-train exact pilot (validation only)
    full_rows = []
    if args.full_exact:
        for cfg in CONFIGS:
            s = sel["per_config_and_fit_n"].get(f"{cfg}|fit_n={max(args.fit_ns)}")
            if not s:
                continue
            est = est_peak_mb(N_tr, D_FEAT, args.cache_mb)
            base = {"run_id": f"B_{cfg}_nFULL_a{s['gamma_scale']}_C{s['C']}", "config": cfg, "fit_n": N_tr,
                    "gamma_scale": s["gamma_scale"], "gamma": s["gamma"], "C": s["C"],
                    "note": "OPTIONAL full-training exact pilot (hyper-parameters of the fit_n=8000 validation winner); validation only"}
            if est > args.max_ram_gb * 1024:
                full_rows.append({**base, "status": "skipped_ram_guard", "skip_reason": f"estimated peak {est:.0f} MB"})
                continue
            log.info("FULL exact pilot %s (N=%d, gamma=%.5f, C=%g, limit %.0fs) ...", cfg, N_tr, s["gamma"], s["C"], args.rbf_max_seconds)
            res = run_bounded(D[cfg]["Xtr"], D[cfg]["ytr"], D[cfg]["Xva"],
                              {"gamma": s["gamma"], "C": s["C"], "seed": args.seed, "cache_mb": args.cache_mb},
                              "ovo_native", status_dir / f"{base['run_id']}.pkl", args.rbf_max_seconds)
            fr = record(base["run_id"], base, res)
            full_rows.append(fr)
            log.info("FULL exact %s -> %s %s", cfg, fr["status"],
                     {k: fr.get(k) for k in ("fit_time_s", "val_scopeB_macro_f1", "n_support_vectors", "child_peak_rss_mb")})
        pd.DataFrame(full_rows).to_csv(out / "results" / "stage_b_full_train_exact_pilot.csv", index=False)

    make_heatmaps(df, out / "figures")
    log.info("Stage B finished: %d/%d grid cells done; ablation rows=%d; full-exact rows=%d", n_done, len(df), len(abl), len(full_rows))
    reg.log(stage="B", status="stage_complete", cells_done=n_done, cells_total=len(df))
    print(f"STAGE_B_SUMMARY fits_done={n_done} of {len(df)} statuses={df['status'].value_counts().to_dict()}")
    return 0


def make_heatmaps(df, fig_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fns = sorted(df["fit_n"].unique())
    fig, axes = plt.subplots(len(CONFIGS), len(fns), figsize=(4.4 * len(fns), 3.6 * len(CONFIGS)), squeeze=False)
    for i, cfg in enumerate(CONFIGS):
        for j, n in enumerate(fns):
            ax = axes[i][j]
            M = np.full((len(GAMMA_SCALES), len(C_VALUES)), np.nan)
            sub = df[(df["config"] == cfg) & (df["fit_n"] == n) & (df["status"] == "done")]
            for _, r in sub.iterrows():
                M[GAMMA_SCALES.index(r["gamma_scale"]), C_VALUES.index(r["C"])] = r["val_scopeB_macro_f1"]
            im = ax.imshow(M, vmin=0, vmax=0.9, cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(C_VALUES)), [f"C={c:g}" for c in C_VALUES])
            ax.set_yticks(range(len(GAMMA_SCALES)), [f"a={a:g}" for a in GAMMA_SCALES])
            for gi in range(len(GAMMA_SCALES)):
                for ci in range(len(C_VALUES)):
                    v = M[gi, ci]
                    ax.text(ci, gi, "not run" if np.isnan(v) else f"{v:.3f}", ha="center", va="center", fontsize=9,
                            color="w" if (np.isnan(v) or v < 0.5) else "k")
            ax.set_title(f"{cfg}, fit_n={n}  (gamma = a/192)", fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Exact RBF SVC (native OvO): validation Scope B macro-F1 (n=4,239)", fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "stage_b_rbf_heatmap_all.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
