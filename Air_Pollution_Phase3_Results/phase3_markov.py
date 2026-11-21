#!/usr/bin/env python3
"""phase3_markov.py -- Phase 3: first-order Markov model + baselines.

    python phase3_markov.py --phase2-dir phase2_output --out phase3_output --seed 42

Only y, target_timestamp, current_state, current_pm25 are loaded (X is never read).
No randomness is used in the pipeline; --seed is recorded for provenance.
"""
from __future__ import annotations

import argparse, hashlib, json, os, platform, statistics, sys, time, zipfile
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

import markov_core as mc

SPLITS = ("train", "validation", "test")
BOUNDS = {"train": ("2013-03-01", "2015-12-31"),
          "validation": ("2016-01-01", "2016-06-30"),
          "test": ("2016-07-01", "2017-02-28")}
EXPECTED = {  # Phase 2 INPUT facts (not Phase 3 results)
    "train": dict(total=24108, known=23976, missing=132, labels=[7872, 5832, 10404]),
    "validation": dict(total=4239, known=4197, missing=42, labels=[1848, 1114, 1277]),
    "test": dict(total=5720, known=5685, missing=35, labels=[2142, 1202, 2376]),
}
MODELS = ("majority", "persistence", "markov")
SCOPES = ("observed_current_only", "full_split_with_prior_fallback")
PRED_COLUMNS = ["target_timestamp", "y", "current_state", "scope_a_eligible", "pred_majority",
                "pred_persistence_scope_b", "pred_markov_scope_b",
                "prob_majority_0", "prob_majority_1", "prob_majority_2",
                "prob_markov_0", "prob_markov_1", "prob_markov_2"]
ROLE = {"train": "in-sample (P fitted here)", "validation": "validation", "test": "test (final, untouched)"}
LAPLACE_ALPHA = 1.0   # fixed in advance for the second-order NLL comparison; not tuned


class Stop(Exception):
    """Hard stop with a clear message (input contract violated)."""


# ----------------------------------------------------------------- helpers
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def array_hash(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def npz_shape(path: Path, key: str):
    """Read an array's shape from the .npy header inside the NPZ WITHOUT loading it."""
    with zipfile.ZipFile(path) as z, z.open(key + ".npy") as f:
        ver = np.lib.format.read_magic(f)
        reader = np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0
        shape, _, dtype = reader(f)
    return tuple(shape), dtype


def jsafe(o):
    if isinstance(o, dict):
        return {str(k): jsafe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsafe(v) for v in o]
    if isinstance(o, np.ndarray):
        if np.iscomplexobj(o):
            return [{"real": float(z.real), "imag": float(z.imag)} for z in o.ravel()]
        return jsafe(o.tolist())
    if isinstance(o, complex):
        return {"real": o.real, "imag": o.imag}
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return "inf" if f == float("inf") else ("-inf" if f == float("-inf") else (None if f != f else f))
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def labelled(M, rows="from_state", cols="to_"):
    names = mc.CLASS_NAMES
    return pd.DataFrame(M, index=[f"{i}_{n}" for i, n in enumerate(names)],
                        columns=[f"{cols}{i}_{n}" for i, n in enumerate(names)]).rename_axis(rows)


def median_time(fn, repeats: int, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    t = []
    for _ in range(repeats):
        t0 = time.perf_counter(); fn(); t.append(time.perf_counter() - t0)
    return float(statistics.median(t))


# ----------------------------------------------------------------- loading / contract
def load_split(d: Path, name: str) -> Dict[str, np.ndarray]:
    p = d / f"{name}_windows.npz"
    if not p.exists():
        raise Stop(f"Required file missing: {p}")
    shape, _ = npz_shape(p, "X")
    with np.load(p, allow_pickle=False) as z:
        missing = [k for k in ("X", "y", "target_timestamp", "current_state", "current_pm25") if k not in z.files]
        if missing:
            raise Stop(f"{p.name}: missing array(s) {missing}")
        out = {k: z[k] for k in ("y", "target_timestamp", "current_state", "current_pm25")}
    n = len(out["y"])
    if shape != (n, 24, 12, 11):
        raise Stop(f"{p.name}: X shape {shape} != ({n},24,12,11)")
    if out["y"].dtype != np.int8 or out["current_state"].dtype != np.int8:
        raise Stop(f"{p.name}: y/current_state must be int8")
    if out["target_timestamp"].dtype != np.dtype("datetime64[ns]"):
        raise Stop(f"{p.name}: target_timestamp must be datetime64[ns]")
    if out["current_pm25"].dtype != np.float32:
        raise Stop(f"{p.name}: current_pm25 must be float32")
    for k in ("target_timestamp", "current_state", "current_pm25"):
        if len(out[k]) != n:
            raise Stop(f"{p.name}: {k} length mismatch")
    out["X_shape"] = shape
    return out


def validate_contract(data) -> Dict[str, object]:
    audit = {}
    for s in SPLITS:
        d, e = data[s], EXPECTED[s]
        y, cs, ts, pm = d["y"], d["current_state"], d["target_timestamp"], d["current_pm25"]
        if not set(np.unique(y)).issubset({0, 1, 2}):
            raise Stop(f"{s}: y outside {{0,1,2}}")
        if not set(np.unique(cs)).issubset({-1, 0, 1, 2}):
            raise Stop(f"{s}: current_state outside {{-1,0,1,2}}")
        known = int((cs >= 0).sum()); missing = int((cs < 0).sum())
        lab = np.bincount(y, minlength=3).tolist()
        if (len(y), known, missing, lab) != (e["total"], e["known"], e["missing"], e["labels"]):
            raise Stop(f"{s}: loaded counts (total={len(y)}, known={known}, missing={missing}, labels={lab}) "
                       f"do not match the Phase 2 contract {e}. Not adjusting silently.")
        derived = np.where(np.isnan(pm), -1, np.where(pm < 35, 0, np.where(pm < 75, 1, 2)))
        if np.any(derived != cs):
            raise Stop(f"{s}: current_state inconsistent with current_pm25 thresholds on "
                       f"{int((derived != cs).sum())} rows")
        audit[s] = {"n_total": len(y), "n_observed_current": known, "n_missing_current": missing,
                    "observation_coverage_pct": 100 * known / len(y), "label_counts": lab}
    audit["split_isolation"] = mc.check_split_isolation(
        {s: data[s]["target_timestamp"] for s in SPLITS}, BOUNDS)
    return audit


# ----------------------------------------------------------------- evaluation
def scope_rows(cs: np.ndarray, scope: str) -> np.ndarray:
    return cs >= 0 if scope == SCOPES[0] else np.ones(len(cs), dtype=bool)


def run_predictors(cs, model_params):
    P, modal, maj, prior = (model_params[k] for k in ("P", "modal", "majority", "prior"))
    n = len(cs)
    return {
        "majority": lambda: mc.predict_majority(n, maj, prior),
        "persistence": lambda: mc.predict_persistence(cs, maj, prior),
        "markov": lambda: mc.predict_markov(cs, P, modal, maj, prior),
    }


def evaluate(data, params, repeats: int, fit_seconds: Dict[str, float]):
    metric_rows, class_rows, cm_rows, change_rows = [], [], [], []
    for s in SPLITS:
        d = data[s]
        y_all, cs_all = d["y"].astype(np.int64), d["current_state"].astype(np.int64)
        n_total, n_obs = len(y_all), int((cs_all >= 0).sum())
        for scope in SCOPES:
            m = scope_rows(cs_all, scope)
            y, cs = y_all[m], cs_all[m]
            n_eval = int(m.sum())
            fns = run_predictors(cs, params)
            for model in MODELS:
                pred, prob = fns[model]()
                n_fb = 0 if (model == "majority" or scope == SCOPES[0]) else int((cs < 0).sum())
                cm = mc.confusion(y, pred)
                cmx = mc.classification_metrics(cm)
                bs = mc.brier_score(prob, y)
                if model == "persistence":
                    ll, nz, note = float("nan"), None, "not reported: persistence one-hot probabilities are uncalibrated"
                    bnote = "one-hot hard decision (uncalibrated) if observed, train prior otherwise"
                else:
                    ll, nz = mc.log_loss_clipped(prob, y)
                    note = f"nats; probabilities clipped to [{mc.EPS_LOGLOSS:g},1] then renormalised"
                    bnote = "probability vector"
                t_inf = median_time(fns[model], repeats)
                metric_rows.append({
                    "split": s, "scope": scope, "model": model, "evaluation_role": ROLE[s],
                    "n_total": n_total, "n_observed_current": n_obs, "n_evaluated": n_eval,
                    "n_excluded": n_total - n_eval, "n_missing_current_in_split": n_total - n_obs,
                    "n_fallback": n_fb,
                    "observation_coverage_pct": 100 * n_obs / n_total,
                    "prediction_coverage_pct": 100 * n_eval / n_total,
                    "fallback_rate_pct": 100 * n_fb / n_eval,
                    "accuracy": cmx["accuracy"], "macro_precision": cmx["macro_precision"],
                    "macro_recall": cmx["macro_recall"], "macro_f1": cmx["macro_f1"],
                    "balanced_accuracy": cmx["balanced_accuracy"],
                    "balanced_accuracy_n_classes_with_support": cmx["balanced_accuracy_n_classes"],
                    "brier_score": bs, "brier_note": bnote,
                    "log_loss": ll, "log_loss_note": note, "n_true_class_zero_prob": nz,
                    "fit_seconds": fit_seconds[model], "inference_seconds_median": t_inf,
                    "n_predictions_timed": n_eval, "timing_repeats": repeats,
                })
                for c in range(3):
                    class_rows.append({"split": s, "scope": scope, "model": model, "class": c,
                                       "class_name": mc.CLASS_NAMES[c], "support": int(cmx["support"][c]),
                                       "n_predicted": int(cmx["n_pred"][c]),
                                       "precision": cmx["precision"][c], "recall": cmx["recall"][c],
                                       "f1": cmx["f1"][c]})
                    for c2 in range(3):
                        cm_rows.append({"split": s, "scope": scope, "model": model,
                                        "true_class": c, "pred_class": c2, "count": int(cm[c, c2])})
                if scope == SCOPES[0]:
                    r = {"split": s, "scope": scope, "model": model,
                         "denominator_note": "rows with observed current_state only"}
                    r.update(mc.change_metrics(y, cs, pred))
                    change_rows.append(r)
    return (pd.DataFrame(metric_rows), pd.DataFrame(class_rows), pd.DataFrame(cm_rows),
            pd.DataFrame(change_rows))


def build_prediction_frame(d, params) -> pd.DataFrame:
    cs = d["current_state"].astype(np.int64)
    n = len(cs)
    P, modal, maj, prior = params["P"], params["modal"], params["majority"], params["prior"]
    pm, prm = mc.predict_majority(n, maj, prior)
    pp, _ = mc.predict_persistence(cs, maj, prior)
    pk, prk = mc.predict_markov(cs, P, modal, maj, prior)
    ts = pd.to_datetime(d["target_timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame({
        "target_timestamp": ts, "y": d["y"].astype(int), "current_state": cs.astype(int),
        "scope_a_eligible": (cs >= 0).astype(int), "pred_majority": pm,
        "pred_persistence_scope_b": pp, "pred_markov_scope_b": pk,
        "prob_majority_0": prm[:, 0], "prob_majority_1": prm[:, 1], "prob_majority_2": prm[:, 2],
        "prob_markov_0": prk[:, 0], "prob_markov_1": prk[:, 1], "prob_markov_2": prk[:, 2]})
    return df[PRED_COLUMNS]


def validate_prediction_frame(df: pd.DataFrame, n_expected: int, n_a_expected: int, n_missing: int, name: str):
    if list(df.columns) != PRED_COLUMNS:
        raise Stop(f"{name}: column order/schema mismatch")
    if len(df) != n_expected or int(df.scope_a_eligible.sum()) != n_a_expected:
        raise Stop(f"{name}: row counts wrong")
    if (df.current_state < 0).sum() != n_missing or ((df.current_state >= 0).astype(int) != df.scope_a_eligible).any():
        raise Stop(f"{name}: missing-state handling / eligibility flag inconsistent")
    ts = pd.to_datetime(df.target_timestamp)
    if not ts.is_monotonic_increasing or ts.duplicated().any():
        raise Stop(f"{name}: chronological order violated")
    for c in ("y", "pred_majority", "pred_persistence_scope_b", "pred_markov_scope_b"):
        if not set(df[c].unique()).issubset({0, 1, 2}):
            raise Stop(f"{name}: {c} outside class labels")
    for m in ("majority", "markov"):
        pr = df[[f"prob_{m}_{i}" for i in range(3)]].to_numpy()
        if not mc.is_row_stochastic(pr, 1e-9):
            raise Stop(f"{name}: prob_{m} not a valid probability matrix")
    # fallback rows must carry the training prior / majority
    fb = df.current_state < 0
    if fb.any():
        if not (df.loc[fb, "pred_markov_scope_b"] == df.loc[fb, "pred_majority"]).all():
            raise Stop(f"{name}: fallback hard prediction is not the training majority")


# ----------------------------------------------------------------- diagnostics
def adequacy_and_drift(data, P):
    """First-order adequacy (timestamp-exact triples) and train-only nonstationarity."""
    res = {}
    series = {s: mc.build_state_series(data[s]["target_timestamp"], data[s]["y"], data[s]["current_state"])
              for s in SPLITS}
    res["series_audit"] = {}
    triples = {}
    for s in SPLITS:
        a, b, c, _ = mc.triples_from_series(series[s]["series"])
        triples[s] = (a, b, c)
        sa, sb, sc = mc.strict_join_triples(data[s]["target_timestamp"], data[s]["y"], data[s]["current_state"])
        res["series_audit"][s] = {
            "n_hours_in_series": int(len(series[s]["series"])),
            "n_hours_observed": int((series[s]["series"] >= 0).sum()),
            "n_conflicts_y_vs_next_current_state": series[s]["n_conflicts"],
            "n_hours_known_only_via_current_state": series[s]["n_hours_from_current_state_only"],
            "n_triples_reconstructed_series": int(len(a)),
            "n_triples_strict_option1_join": int(len(sa)),
        }
        if series[s]["n_conflicts"]:
            raise Stop(f"{s}: y(t-1) disagrees with current_state(t) in {series[s]['n_conflicts']} rows")
    # --- second order (train fit, validation NLL; test untouched)
    c2 = mc.counts3(*triples["train"])
    c1_tr = c2.sum(axis=0)                      # first-order counts on the SAME triple set
    n_ctx = c2.sum(axis=2)
    p2_mle = np.divide(c2, n_ctx[..., None], out=np.zeros(c2.shape), where=n_ctx[..., None] > 0)
    p1_mle = c1_tr / c1_tr.sum(axis=1, keepdims=True)
    rows = []
    for h2 in range(3):
        for h1 in range(3):
            tvd = 0.5 * float(np.abs(p2_mle[h2, h1] - p1_mle[h1]).sum()) if n_ctx[h2, h1] else float("nan")
            rows.append({"S_t-2": h2, "S_t-1": h1, "n_train_triples": int(n_ctx[h2, h1]),
                         **{f"P2_to_{j}": p2_mle[h2, h1, j] for j in range(3)},
                         **{f"P1_to_{j}": p1_mle[h1, j] for j in range(3)}, "tvd_second_vs_first": tvd,
                         "low_support_lt_100": bool(n_ctx[h2, h1] < 100)})
    ad_table = pd.DataFrame(rows)
    a = LAPLACE_ALPHA
    p2_s = (c2 + a) / (n_ctx[..., None] + 3 * a)
    p1_s = (c1_tr + a) / (c1_tr.sum(axis=1, keepdims=True) + 3 * a)
    va, vb, vc = triples["validation"]
    nll2 = float(-np.mean(np.log(p2_s[va, vb, vc])))
    nll1 = float(-np.mean(np.log(p1_s[vb, vc])))
    Pp = np.clip(P, mc.EPS_LOGLOSS, 1.0); Pp = Pp / Pp.sum(axis=1, keepdims=True)
    nll_primary = float(-np.mean(np.log(Pp[vb, vc])))
    # likelihood ratio on TRAIN triples (descriptive; serial dependence invalidates iid p-values)
    from scipy.stats import chi2
    with np.errstate(divide="ignore", invalid="ignore"):
        ll2 = float(np.nansum(np.where(c2 > 0, c2 * np.log(p2_mle), 0.0)))
        ll1 = float(np.nansum(np.where(c1_tr > 0, c1_tr * np.log(p1_mle), 0.0)))
    df_lr = 2 * (int((n_ctx > 0).sum()) - int((c1_tr.sum(axis=1) > 0).sum()))
    G = 2 * (ll2 - ll1)
    ntr = int(c2.sum())
    res["adequacy"] = {
        "method": "timestamp-exact triples from the observed hourly series reconstructed from y(t) and current_state(t) "
                  "(never from shifted filtered arrays; X never used); strict Option-1 join counts reported for audit",
        "n_train_triples": ntr, "n_validation_triples": int(len(va)),
        "laplace_alpha_fixed_in_advance": a,
        "validation_mean_nll_nats": {"first_order_laplace_same_triple_fit": nll1,
                                     "second_order_laplace": nll2,
                                     "primary_P_mle_clipped_eps": nll_primary},
        "second_minus_first_nll_validation": nll2 - nll1,
        "train_loglik_first_order": ll1, "train_loglik_second_order": ll2,
        "likelihood_ratio_G": G, "lr_df": df_lr,
        "lr_pvalue_nominal_iid": float(chi2.sf(G, df_lr)),
        "aic_first": 2 * 6 - 2 * ll1, "aic_second": 2 * 18 - 2 * ll2,
        "bic_first": 6 * np.log(ntr) - 2 * ll1, "bic_second": 18 * np.log(ntr) - 2 * ll2,
        "caveat": "Hourly states are serially dependent; nominal iid chi-square p-values are descriptive only. "
                  "A formal Markov-property claim is NOT made.",
    }
    # --- nonstationarity on TRAIN only
    d = data["train"]
    ts = pd.to_datetime(d["target_timestamp"])
    cs, y = d["current_state"].astype(np.int64), d["y"].astype(np.int64)
    groups = {"year": ts.year.to_numpy(), "season": ts.month.map(mc.SEASON_OF_MONTH).to_numpy()}
    Cglob = mc.counts_from_pairs(cs, y)
    Pglob = P
    tab = []
    summ = []
    for gtype, g in groups.items():
        for lab in sorted(set(g.tolist())):
            m = g == lab
            C = mc.counts_from_pairs(cs[m], y[m])
            n = C.sum(axis=1)
            Pg = np.divide(C, n[:, None], out=np.full((3, 3), np.nan), where=n[:, None] > 0)
            for i in range(3):
                tvd = 0.5 * float(np.nansum(np.abs(Pg[i] - Pglob[i]))) if n[i] else float("nan")
                summ.append({"group_type": gtype, "group": str(lab), "from_state": i, "row_support": int(n[i]),
                             "tvd_vs_pooled_train_row": tvd, "low_support_lt_100": bool(n[i] < 100)})
                for j in range(3):
                    tab.append({"group_type": gtype, "group": str(lab), "from_state": i, "to_state": j,
                                "count": int(C[i, j]), "row_support": int(n[i]), "prob": Pg[i, j],
                                "pooled_train_prob": Pglob[i, j], "diff_vs_pooled": Pg[i, j] - Pglob[i, j],
                                "low_support_lt_100": bool(n[i] < 100)})
    res["nonstat_table"] = pd.DataFrame(tab)
    res["nonstat_summary"] = pd.DataFrame(summ)
    res["season_definition"] = "meteorological by calendar month of target timestamp: DJF=Dec-Feb, MAM=Mar-May, JJA=Jun-Aug, SON=Sep-Nov; pooled over training years; year = calendar year of target timestamp (2013 starts in March)"
    res["adequacy_table"] = ad_table
    # empirical run lengths (train, observed contiguous segments)
    rl = mc.run_lengths(series["train"]["series"])
    emp = {}
    for st in range(3):
        allr = [l for (s_, l, c) in rl if s_ == st]
        unc = [l for (s_, l, c) in rl if s_ == st and not c]
        emp[st] = {"n_runs_all": len(allr), "mean_all": float(np.mean(allr)) if allr else float("nan"),
                   "n_runs_uncensored": len(unc), "mean_uncensored": float(np.mean(unc)) if unc else float("nan")}
    res["empirical_runs_train"] = emp
    res["train_series"] = series["train"]["series"]
    return res


# ----------------------------------------------------------------- figures
def make_figures(out: Path, P, counts, spectral, drift, metrics_df, cm_df, data, pi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Circle
    fig_dir = out / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    names = [f"{i} {n}" for i, n in enumerate(mc.CLASS_NAMES)]

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(P, vmin=0, vmax=1, cmap="viridis")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{P[i, j]:.3f}\n(n={counts[i, j]})", ha="center", va="center",
                    color="white" if P[i, j] < 0.6 else "black", fontsize=9)
    ax.set_xticks(range(3), names); ax.set_yticks(range(3), names)
    ax.set_xlabel("next-hour state S_t"); ax.set_ylabel("current state S_(t-1)")
    ax.set_title("Estimated transition matrix P (MLE)\nTRAIN 2013-03..2015-12, pairs with observed S_(t-1) (n=23,976)", fontsize=9)
    fig.colorbar(im, label="P(S_t=j | S_(t-1)=i)"); fig.tight_layout()
    fig.savefig(fig_dir / "fig1_transition_heatmap_train.png", dpi=160); plt.close(fig)

    # transition graph
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    pos = {0: (-1.0, -0.6), 1: (1.0, -0.6), 2: (0.0, 1.0)}
    cols = {0: "#4c9f70", 1: "#e0b040", 2: "#c0504d"}
    for i, (x, y) in pos.items():
        ax.add_patch(Circle((x, y), 0.22, color=cols[i], ec="k", zorder=3))
        ax.text(x, y, f"{i}\n{mc.CLASS_NAMES[i]}", ha="center", va="center", fontsize=8, zorder=4)
    for i in range(3):
        for j in range(3):
            p = P[i, j]
            if p <= 0:
                continue
            (x1, y1), (x2, y2) = pos[i], pos[j]
            if i == j:
                u = np.array({0: (-0.86, -0.51), 1: (0.86, -0.51), 2: (0.0, 1.0)}[i])
                cx, cy = x1 + 0.33 * u[0], y1 + 0.33 * u[1]
                ax.add_patch(Circle((cx, cy), 0.16, fill=False, lw=0.6 + 3 * p, ec="0.25", zorder=1))
                ax.text(cx + 0.30 * u[0], cy + 0.30 * u[1], f"{p:.3f}", ha="center", va="center", fontsize=8)
            else:
                ar = FancyArrowPatch((x1, y1), (x2, y2), connectionstyle="arc3,rad=0.22", arrowstyle="-|>",
                                     mutation_scale=14, lw=0.5 + 5 * p, shrinkA=22, shrinkB=22, color="0.25", zorder=2)
                ax.add_patch(ar)
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                nx, ny = -(y2 - y1), (x2 - x1)
                nrm = np.hypot(nx, ny)
                ax.text(mx - 0.22 * nx / nrm * 0.9, my - 0.22 * ny / nrm * 0.9, f"{p:.3f}", fontsize=8,
                        ha="center", va="center", bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.8))
    ax.set_xlim(-2, 2); ax.set_ylim(-1.7, 2.0); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Directed transition graph, first-order Markov model\nTRAIN estimates; edge label and width = P(i->j); zero edges omitted", fontsize=9)
    fig.tight_layout(); fig.savefig(fig_dir / "fig2_transition_graph_train.png", dpi=160); plt.close(fig)

    # confusion matrices (val, test), both scopes, three models
    for s in ("validation", "test"):
        fig, axes = plt.subplots(2, 3, figsize=(11, 7))
        for r, scope in enumerate(SCOPES):
            for c, model in enumerate(MODELS):
                sub = cm_df[(cm_df.split == s) & (cm_df.scope == scope) & (cm_df.model == model)]
                M = sub.sort_values(["true_class", "pred_class"])["count"].to_numpy().reshape(3, 3)
                Mn = M / np.maximum(M.sum(axis=1, keepdims=True), 1)
                ax = axes[r, c]
                ax.imshow(Mn, vmin=0, vmax=1, cmap="Blues")
                for i in range(3):
                    for j in range(3):
                        ax.text(j, i, f"{M[i, j]}\n{100 * Mn[i, j]:.1f}%", ha="center", va="center",
                                color="white" if Mn[i, j] > 0.55 else "black", fontsize=8)
                ax.set_xticks(range(3), names, fontsize=8); ax.set_yticks(range(3), names, fontsize=8)
                ax.set_xlabel("predicted state"); ax.set_ylabel("true state S_t")
                row = metrics_df[(metrics_df.split == s) & (metrics_df.scope == scope) & (metrics_df.model == model)].iloc[0]
                ax.set_title(f"{model} | {'Scope A observed-current' if r == 0 else 'Scope B full + fallback'}\n"
                             f"n={row.n_evaluated}, acc={row.accuracy:.3f}, macro-F1={row.macro_f1:.3f}", fontsize=8)
        fig.suptitle(f"Confusion matrices (counts and row-%), {s.upper()} split", fontsize=11)
        fig.tight_layout(); fig.savefig(fig_dir / f"fig3_confusion_{s}.png", dpi=150); plt.close(fig)

    # empirical fractions vs pi
    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.2
    for k, s in enumerate(SPLITS):
        frac = np.bincount(data[s]["y"], minlength=3) / len(data[s]["y"])
        ax.bar(np.arange(3) + (k - 1.5) * w, frac, w, label=f"{s} empirical y fractions (descriptive)")
    ax.bar(np.arange(3) + 1.5 * w, pi, w, label="stationary pi of TRAIN P (model-derived)", hatch="//", color="0.6")
    ax.set_xticks(range(3), names); ax.set_ylabel("fraction of hours"); ax.legend(fontsize=7)
    ax.set_title("Empirical state fractions vs stationary distribution (descriptive; pi is not fitted to val/test)", fontsize=9)
    fig.tight_layout(); fig.savefig(fig_dir / "fig4_state_fractions_vs_stationary.png", dpi=160); plt.close(fig)

    # run lengths
    fig, ax = plt.subplots(figsize=(6.5, 4))
    expd = mc.expected_run_lengths(P)
    emp = drift["empirical_runs_train"]
    ax.bar(np.arange(3) - 0.2, [min(e, 1e3) for e in expd], 0.4, label="model-implied 1/(1-P_ii)")
    ax.bar(np.arange(3) + 0.2, [emp[i]["mean_all"] for i in range(3)], 0.4, label="empirical mean run (TRAIN, incl. edge-censored)")
    ax.set_xticks(range(3), names); ax.set_ylabel("run length (hours)"); ax.legend(fontsize=8)
    ax.set_title("Regime run lengths: model-implied vs observed (TRAIN)", fontsize=9)
    fig.tight_layout(); fig.savefig(fig_dir / "fig5_run_lengths_train.png", dpi=160); plt.close(fig)

    # drift
    nt = drift["nonstat_table"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    for ax, gtype in zip(axes, ("year", "season")):
        sub = nt[nt.group_type == gtype]
        groups = sorted(sub.group.unique(), key=lambda g: (["DJF", "MAM", "JJA", "SON"].index(g) if gtype == "season" else int(g)))
        labels = [f"{i}->{j}" for i in range(3) for j in range(3)]
        wb = 0.8 / len(groups)
        for k, g in enumerate(groups):
            vals = sub[sub.group == g].sort_values(["from_state", "to_state"])["prob"].to_numpy()
            ax.bar(np.arange(9) + (k - (len(groups) - 1) / 2) * wb, vals, wb, label=str(g))
        ax.scatter(np.arange(9), P.ravel(), marker="_", s=300, c="k", label="pooled TRAIN P", zorder=5)
        ax.set_xticks(range(9), labels); ax.set_xlabel("transition i->j"); ax.legend(fontsize=7)
        ax.set_title(f"TRAIN-only transition probabilities by {gtype}", fontsize=9)
    axes[0].set_ylabel("P(S_t=j | S_(t-1)=i)")
    fig.tight_layout(); fig.savefig(fig_dir / "fig6_train_drift_transitions.png", dpi=150); plt.close(fig)


# ----------------------------------------------------------------- report
def fmt_mat(M, fmt="{:.4f}"):
    return "\n".join("    [" + ", ".join(fmt.format(v) for v in row) + "]" for row in M)


def write_report(out: Path, ctx):
    P, counts, P2, st, md = ctx["P"], ctx["counts"], ctx["P2"], ctx["spec"], ctx["metrics"]
    ad, nsum = ctx["drift"]["adequacy"], ctx["drift"]["nonstat_summary"]
    L = []
    A = L.append
    A("PHASE 3 -- FIRST-ORDER MARKOV MODEL AND SEQUENTIAL BASELINES: RESULTS AND MATHEMATICS")
    A("Target: Aotizhongxin hourly PM2.5 state (research bins, NOT official AQI): 0 low <35, 1 moderate 35-<75, 2 high >=75 ug/m3.")
    A("Everything in sections 3-9 is MEASURED by running phase3_markov.py on the supplied Phase 2 data.")
    A("Section 1-2 and the hand example (section 2) are ILLUSTRATIVE / derivations, not Beijing results.\n")
    A("=" * 78); A("1. MATHEMATICAL DEFINITIONS"); A("=" * 78)
    A("""S_t in {0,1,2}. For a labelled row predicting hour t: current_state = S_(t-1), y = S_t.
Counts      C_ij = #{train rows : current_state=i, y=j, i>=0}
MLE         P_ij = C_ij / sum_j C_ij      (maximises prod P_ij^C_ij subject to row sums 1; Lagrange: C_ij/lambda_i)
Two-step    (P^2)_ij = sum_k P_ik P_kj    (Chapman-Kolmogorov; a MODEL consequence, not measured 2-hour accuracy)
Stationary  pi P = pi, pi>=0, sum pi = 1  <=>  (P^T - I) pi = 0 with the normalisation row appended (solved by least squares;
            rank of the (k+1) x k system = k means the solution is unique).
            Unique if the chain is irreducible (one communicating class). Convergence pi_0 P^n -> pi needs aperiodicity too.
Spectral gap gamma = 1 - max|lambda| over non-trivial eigenvalues; read as a mixing-rate indicator ONLY if the chain is
            irreducible, aperiodic and eigenvalue 1 is simple (otherwise reported as 'not interpreted').
Run length  E[run | state i] = 1/(1-P_ii) for P_ii<1 under the time-homogeneous first-order model (geometric sojourn).
Predictors  majority: constant argmax train y-counts (ties -> smallest index).  persistence: S_hat_t = S_(t-1).
            Markov (modal): S_hat_t = argmax_j P[S_(t-1), j] with LOCKED tie rule: if current state i is in the argmax set, predict i;
            else predict min(argmax set). Ties are decided by exact integer comparison of counts (same row denominator).
            Majority vs persistence vs modal Markov: majority ignores the state; persistence is the Markov rule restricted to
            'always stay'; the modal Markov rule only departs from persistence when some other state has a strictly higher
            estimated next-hour probability than staying.
Metrics     confusion matrix M (rows true, cols predicted); Precision_c = M_cc/sum_i M_ic; Recall_c = M_cc/sum_j M_cj;
            F1_c = 2PR/(P+R); macro-F1 = mean_c F1_c; balanced accuracy = mean recall over classes with support>0 (zero_division=0).
            Brier = mean_i sum_c (p_ic - 1[y_i=c])^2 (unnormalised, range 0..2).  Log loss uses clip eps=1e-12 then renormalise
            (fixed, never tuned); not reported for persistence (one-hot, uncalibrated).\n""")
    A("=" * 78); A("2. HAND-WORKED ILLUSTRATIVE EXAMPLE (NOT BEIJING DATA)"); A("=" * 78)
    ex = ctx["example"]
    A("Sequence L,L,M,M,H,H,M,L,L = 0,0,1,1,2,2,1,0,0  -> 8 one-hour transitions.")
    A("C =\n" + fmt_mat(ex["C"], "{:d}"))
    A("P = [[2/3,1/3,0],[1/3,1/3,1/3],[0,1/2,1/2]] =\n" + fmt_mat(ex["P"]))
    A("P^2 = [[5/9,1/3,1/9],[1/3,7/18,5/18],[1/6,5/12,5/12]] =\n" + fmt_mat(ex["P2"]))
    A("e.g. (P^2)_00 = 2/3*2/3 + 1/3*1/3 + 0*0 = 4/9+1/9 = 5/9.")
    A("pi = [3/8,3/8,1/4]: check pi P col0 = 3/8*2/3+3/8*1/3+0 = 1/4+1/8 = 3/8; col2 = 3/8*1/3+1/4*1/2 = 1/8+1/8 = 1/4.")
    A("Eigenvalues (numerical): " + ", ".join(f"{v.real:.8f}" for v in ex["eig"]) + "  -> illustrative gap ~ %.8f" % ex["gap"])
    A("Tie-break: state M (1) has equal row (1/3,1/3,1/3) -> predict 1 (current state in tie set); state H (2) ties M/H -> predict 2.")
    A("Persistence metrics on its 8 pairs: accuracy 4/8; M = [[2,1,0],[1,1,1],[0,1,1]]; precision=recall=F1=[2/3,1/3,1/2];")
    A("macro-F1 = (2/3+1/3+1/2)/3 = 1/2; balanced accuracy = 1/2.  (code reproduces: acc=%.3f macro-F1=%.3f bal=%.3f)\n"
      % (ex["acc"], ex["maf1"], ex["bal"]))
    A("=" * 78); A("3. INPUT AUDIT (MEASURED)"); A("=" * 78)
    for s in SPLITS:
        a = ctx["audit"][s]
        A(f"{s:11s} total={a['n_total']} observed_current={a['n_observed_current']} missing_current={a['n_missing_current']} "
          f"observation_coverage={a['observation_coverage_pct']:.2f}% labels[low,mod,high]={a['label_counts']}")
    A("All loaded counts equal the Phase 2 contract; split isolation / strict timestamp order verified.")
    A("Training prior uses all 24,108 y labels (independent of the missing-current mask); transition counts use 23,976 complete pairs.\n")
    A("=" * 78); A("4. TRAIN-ONLY MARKOV ESTIMATES (MEASURED)"); A("=" * 78)
    A(f"C (rows=current state 0,1,2; cols=next state), C.sum()={int(counts.sum())}, row sums={counts.sum(1).tolist()}:\n" + fmt_mat(counts, "{:d}"))
    A("P (MLE):\n" + fmt_mat(P, "{:.6f}"))
    A("P^2:\n" + fmt_mat(P2, "{:.6f}"))
    A(f"row-sum max error P: {np.abs(P.sum(1)-1).max():.2e}; P^2: {np.abs(P2.sum(1)-1).max():.2e}; min entries P={P.min():.6f}, P^2={P2.min():.6f}")
    A(f"modal prediction by current state [0,1,2] = {ctx['modal'].tolist()}; tie sets = {ctx['ties']}; majority label = {ctx['majority']}; "
      f"train prior = {np.round(ctx['prior'], 6).tolist()}\n")
    A("=" * 78); A("5. STATIONARY DISTRIBUTION, SPECTRUM, STRUCTURE (MEASURED)"); A("=" * 78)
    A(f"pi = {np.round(st['pi'], 6).tolist()}  residual(inf-norm)={st['stationary_residual_inf_norm']:.2e}  sum error={st['pi_sum_error']:.2e}  min={st['pi_min']:.3e}")
    A(f"constraint-system rank={st['constraint_matrix_rank']} (3 => unique), condition number={st['constraint_matrix_condition']:.3f}; "
      f"left-eigenvector cross-check max diff={st['pi_vs_eigenvector_max_abs_diff']:.2e}")
    A("eigenvalues: " + ", ".join(f"{z.real:.6f}{z.imag:+.6f}i" for z in st["eigenvalues"]) + f"  (multiplicity of 1 = {st['algebraic_multiplicity_of_1']})")
    A(f"irreducible={st['irreducible']} (positive-edge reachability closure); per-state periods={st['state_periods']} ; aperiodic={st['aperiodic_given_irreducible']}")
    A(f"spectral gap: {st['spectral_gap'] if st['spectral_gap'] is not None else 'n/a'} -- {st['spectral_gap_status']}")
    A("Spectral gap is a property of the FITTED model; it is not a physical pollution-clearing time.")
    er = mc.expected_run_lengths(P)
    emp = ctx["drift"]["empirical_runs_train"]
    for i in range(3):
        A(f"state {i} {mc.CLASS_NAMES[i]}: model-implied mean run = {er[i]:.3f} h; observed TRAIN mean run (all runs, edge-censored included) = "
          f"{emp[i]['mean_all']:.3f} h over {emp[i]['n_runs_all']} runs; uncensored runs only = {emp[i]['mean_uncensored']:.3f} h over {emp[i]['n_runs_uncensored']}")
    A("Empirical class fractions (descriptive, not used to fit pi): " + "; ".join(
        f"{s}={np.round(np.bincount(ctx['data'][s]['y'], minlength=3)/len(ctx['data'][s]['y']), 4).tolist()}" for s in SPLITS) + "\n")
    A("=" * 78); A("6. BASELINE METRICS (MEASURED; Scope A = observed current only, Scope B = all rows + train-prior fallback)"); A("=" * 78)
    show = md[["split", "scope", "model", "n_total", "n_observed_current", "n_evaluated", "n_fallback",
               "observation_coverage_pct", "prediction_coverage_pct", "fallback_rate_pct", "accuracy", "macro_precision",
               "macro_recall", "macro_f1", "balanced_accuracy", "brier_score", "log_loss"]].copy()
    A(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    A("\nNOTE: Scope B prediction coverage is 100% of rows, but observation coverage of the previous-hour state is below 100%;")
    A("the two are different quantities. Train rows are IN-SAMPLE for the Markov model. Test rows were evaluated once, with all parameters fixed.")
    A("Timing (median of batched runs; see markov_baseline_metrics.csv columns fit_seconds / inference_seconds_median).")
    A("\nClasswise metrics: see markov_classwise_metrics.csv.  Confusion matrices: markov_confusion_matrices.csv and figures/fig3_*.\n")
    A("--- Regime-change diagnostics (Scope A rows only; denominators in CSV) ---")
    ch = ctx["change"]
    A(ch.drop(columns=["scope", "denominator_note"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    A("Scope B diagnostics are not computed for unknown-current rows (a 'change' is undefined); on observed-current rows they coincide with Scope A.\n")
    A("=" * 78); A("7. MARKOV ADEQUACY (MEASURED, TRAIN fit / VALIDATION NLL; test untouched)"); A("=" * 78)
    A(ad["method"])
    A("Triple audit: " + json.dumps(jsafe(ctx["drift"]["series_audit"])))
    A(ctx["drift"]["adequacy_table"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    nl = ad["validation_mean_nll_nats"]
    A(f"Validation mean NLL (nats, Laplace alpha={ad['laplace_alpha_fixed_in_advance']} fixed in advance): first-order={nl['first_order_laplace_same_triple_fit']:.5f}, "
      f"second-order={nl['second_order_laplace']:.5f}, difference (2nd-1st)={ad['second_minus_first_nll_validation']:.5f}; primary P (MLE, clipped)={nl['primary_P_mle_clipped_eps']:.5f}")
    A(f"Train LR statistic G={ad['likelihood_ratio_G']:.2f} on df={ad['lr_df']} (nominal p={ad['lr_pvalue_nominal_iid']:.3g}); "
      f"AIC first/second = {ad['aic_first']:.1f}/{ad['aic_second']:.1f}; BIC first/second = {ad['bic_first']:.1f}/{ad['bic_second']:.1f}")
    A("Caveat: " + ad["caveat"] + "\n")
    A("=" * 78); A("8. NONSTATIONARITY ON TRAIN SUBPERIODS (MEASURED, descriptive)"); A("=" * 78)
    A(ctx["drift"]["season_definition"])
    A(nsum.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    A("Full counts/probabilities: markov_train_subperiod_transitions.csv. Rows with support <100 are flagged; no strong claims from them.\n")
    A("=" * 78); A("9. LEAKAGE / REPRODUCIBILITY AUDIT (MEASURED)"); A("=" * 78)
    for k, v in ctx["hashes"].items():
        A(f"sha256(P) {k}: {v}")
    A(f"P hash identical across fit / validation / test / end: {ctx['hash_equal']}")
    A("No fused Markov/HoSVD/PCA/RFF/SVM models, no imputation of current_state, X never loaded, no RNG used (seed recorded only).")
    A("Limitations: raw PRSA CSV not supplied, so adequacy uses the timestamp-exact reconstruction (not Option 2); block bootstrap not implemented (optional).")
    (out / "PHASE3_RESULTS_AND_MATH.txt").write_text("\n".join(L), encoding="utf-8")


# ----------------------------------------------------------------- zip
def make_zip(root: Path, out: Path):
    out = Path(out).resolve()
    zpath = root / "Air_Pollution_Phase3_Results.zip"
    files = [root / f for f in ("phase3_markov.py", "markov_core.py", "requirements.txt", "README.md", "pytest.ini")]
    files += sorted((root / "tests").glob("*.py"))
    files += sorted(p for p in out.rglob("*") if p.is_file())
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            if f.exists():
                z.write(f, f.relative_to(root))
    return zpath


# ----------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase2-dir", default="phase2_output")
    ap.add_argument("--out", default="phase3_output")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timing-repeats", type=int, default=25)
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--zip-only", action="store_true", help="only (re)build the results ZIP from existing outputs")
    a = ap.parse_args(argv)
    root = Path(__file__).resolve().parent
    out = Path(a.out)
    if a.zip_only:
        print("wrote", make_zip(root, out)); return 0
    t_run = time.perf_counter()
    np.random.seed(a.seed)
    d2 = Path(a.phase2_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = {s: load_split(d2, s) for s in SPLITS}
    audit = validate_contract(data)
    print("[ok] Phase 2 contract verified:", {s: audit[s]["n_total"] for s in SPLITS})

    # ---- fit once, TRAIN only
    tr = data["train"]
    cs_tr, y_tr = tr["current_state"].astype(np.int64), tr["y"].astype(np.int64)
    class_counts = np.bincount(y_tr, minlength=3)

    def fit():
        C = mc.counts_from_pairs(cs_tr, y_tr)
        maj = mc.majority_label(class_counts)
        P, rown, prior, fb = mc.fit_markov(C, class_counts)
        modal, ties = mc.modal_rule(C, maj)
        return C, maj, P, rown, prior, fb, modal, ties

    fit_markov_s = median_time(fit, 15)
    fit_major_s = median_time(lambda: mc.majority_label(np.bincount(y_tr, minlength=3)), 15)
    C, maj, P, rown, prior, fb, modal, ties = fit()
    if C.sum() != EXPECTED["train"]["known"] or C.sum() == EXPECTED["train"]["total"]:
        raise Stop(f"C.sum()={C.sum()} but 23976 complete training pairs expected (not 24108)")
    if np.any(rown == 0):
        print(f"[WARN] zero-support transition row(s) {fb}: using TRAIN prior (fallback)")
    P2 = P @ P
    if not (mc.is_row_stochastic(P) and mc.is_row_stochastic(P2)):
        raise Stop("P or P^2 not row-stochastic within 1e-12")
    P.setflags(write=False); modal.setflags(write=False)
    params = {"P": P, "modal": modal, "majority": maj, "prior": prior}
    hashes = {"after_fit": array_hash(P)}
    fit_seconds = {"majority": fit_major_s, "persistence": 0.0, "markov": fit_markov_s}

    # ---- evaluation (validation then test; hash checked between stages)
    metrics, classwise, cm_df, change = evaluate(data, params, a.timing_repeats, fit_seconds)
    hashes["after_evaluation_all_splits"] = array_hash(P)

    # ---- spectral
    spec = mc.stationary_analysis(P)
    drift = adequacy_and_drift(data, P)
    hashes["after_diagnostics"] = array_hash(P)

    # ---- acceptance checks
    if len(metrics) != 18:
        raise Stop("expected 18 metric rows")
    for s in SPLITS:
        for sc in SCOPES:
            sub = metrics[(metrics.split == s) & (metrics.scope == sc)]
            if sub.n_evaluated.nunique() != 1:
                raise Stop(f"{s}/{sc}: models evaluated on different rows")
    fbk = metrics[(metrics.scope == SCOPES[1]) & (metrics.model != "majority")].groupby("split").n_fallback.first().to_dict()
    if fbk != {s: EXPECTED[s]["missing"] for s in SPLITS}:
        raise Stop(f"fallback counts {fbk} != expected")
    if metrics[(metrics.scope == SCOPES[0])].n_fallback.sum() != 0 or metrics[metrics.model == "majority"].n_fallback.sum() != 0:
        raise Stop("unexpected fallback in Scope A or majority")
    hash_equal = len(set(hashes.values())) == 1
    if not hash_equal:
        raise Stop("P changed between stages (refit detected)")

    # ---- write tables
    labelled(C, "current_state", "next_").to_csv(out / "markov_transition_counts.csv")
    labelled(P, "current_state", "next_").to_csv(out / "markov_transition_matrix.csv", float_format="%.12f")
    labelled(P2, "current_state", "next2_").to_csv(out / "markov_two_step_matrix.csv", float_format="%.12f")
    metrics.to_csv(out / "markov_baseline_metrics.csv", index=False)
    classwise.to_csv(out / "markov_classwise_metrics.csv", index=False)
    cm_df.to_csv(out / "markov_confusion_matrices.csv", index=False)
    change.to_csv(out / "markov_change_metrics.csv", index=False)
    drift["adequacy_table"].to_csv(out / "markov_adequacy_second_order.csv", index=False)
    drift["nonstat_table"].to_csv(out / "markov_train_subperiod_transitions.csv", index=False)
    drift["nonstat_summary"].to_csv(out / "markov_train_subperiod_summary.csv", index=False)
    for s in ("validation", "test"):
        df = build_prediction_frame(data[s], params)
        e = EXPECTED[s]
        validate_prediction_frame(df, e["total"], e["known"], e["missing"], f"predictions_{s}")
        df.to_csv(out / f"markov_predictions_{s}.csv", index=False, float_format="%.12f")
        chk = pd.read_csv(out / f"markov_predictions_{s}.csv")
        validate_prediction_frame(chk.assign(target_timestamp=chk.target_timestamp.astype(str)), e["total"], e["known"], e["missing"], f"reloaded_{s}")

    spec_json = {k: v for k, v in spec.items() if k != "reachability_matrix"}
    spec_json["reachability_matrix"] = spec["reachability_matrix"]
    spec_json["expected_run_length_hours"] = mc.expected_run_lengths(P)
    spec_json["empirical_run_lengths_train"] = drift["empirical_runs_train"]
    spec_json["tolerance"] = {"row_stochastic_atol": mc.ROW_TOL, "rtol": 0}
    spec_json["P2_row_sum_max_error"] = float(np.abs(P2.sum(1) - 1).max())
    spec_json["P2_min_entry"] = float(P2.min())
    spec_json["P_min_entry"] = float(P.min())
    spec_json["empirical_class_fractions_descriptive_only"] = {
        s: (np.bincount(data[s]["y"], minlength=3) / len(data[s]["y"])).tolist() for s in SPLITS}
    spec_json["adequacy"] = drift["adequacy"]
    (out / "markov_stationary_spectral.json").write_text(json.dumps(jsafe(spec_json), indent=2))

    np.savez(out / "markov_model_parameters.npz", P=P, P2=P2, counts=C, train_label_counts=class_counts,
             prior=prior, class_order=np.array([0, 1, 2]), class_names=np.array(mc.CLASS_NAMES),
             majority_label=np.int64(maj), modal_prediction_by_current_state=modal,
             tie_break_rule=np.array("tie set A=argmax_j P[i,j] by exact integer count equality; if i in A predict i else min(A); "
                                     "missing current state -> train majority class and train prior"))

    # ---- hand example (illustrative)
    seq = [0, 0, 1, 1, 2, 2, 1, 0, 0]
    Ce = mc.counts_from_sequence(seq); Pe, _, pe, _ = mc.fit_markov(Ce, np.array([1, 1, 1]))
    we = mc.stationary_analysis(Pe)
    cm_e = mc.confusion(np.array(seq[1:]), np.array(seq[:-1])); me = mc.classification_metrics(cm_e)
    example = {"C": Ce, "P": Pe, "P2": Pe @ Pe, "eig": we["eigenvalues"], "gap": we["spectral_gap"],
               "acc": me["accuracy"], "maf1": me["macro_f1"], "bal": me["balanced_accuracy"]}

    make_figures(out, P, C, spec, drift, metrics, cm_df, data, spec["pi"])
    write_report(out, {"P": P, "counts": C, "P2": P2, "spec": spec, "metrics": metrics, "drift": drift,
                       "audit": audit, "modal": modal, "ties": ties, "majority": maj, "prior": prior,
                       "data": data, "change": change, "example": example, "hashes": hashes,
                       "hash_equal": hash_equal})

    meta = {
        "inputs": {f"{s}_windows.npz": {"sha256": sha256(d2 / f"{s}_windows.npz"), "X_shape": list(data[s]["X_shape"])} for s in SPLITS},
        "phase2_summary_sha256": sha256(d2 / "phase2_summary.json") if (d2 / "phase2_summary.json").exists() else None,
        "split_checks": audit, "seed": a.seed, "rng_used": False,
        "rules": {"state_thresholds_ug_m3": {"low": "<35", "moderate": "[35,75)", "high": ">=75"},
                  "markov_tie_break": "i in argmax set -> i else min(argmax set); exact integer-count equality",
                  "missing_current_fallback": "train majority label / train label prior",
                  "train_prior_source": "all 24,108 train y labels", "transition_counts_source": "23,976 train pairs with observed current_state",
                  "log_loss_eps": mc.EPS_LOGLOSS, "laplace_alpha_adequacy": LAPLACE_ALPHA,
                  "brier": "unnormalised sum over 3 classes, mean over rows",
                  "balanced_accuracy": "mean recall over classes with support>0; n classes reported"},
        "P_sha256": hashes, "P_unchanged_across_stages": hash_equal,
        "timing": {"fit_seconds": fit_seconds, "timing_repeats": a.timing_repeats, "warmup": 2,
                   "statistic": "median of batched calls (hard labels + probability rows)",
                   "total_run_seconds": time.perf_counter() - t_run},
        "environment": {"python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
                        "machine": platform.machine(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    import scipy, sklearn, matplotlib
    meta["environment"].update(scipy=scipy.__version__, sklearn=sklearn.__version__, matplotlib=matplotlib.__version__)
    (out / "phase3_metadata.json").write_text(json.dumps(jsafe(meta), indent=2))
    if not a.no_zip:
        print("wrote", make_zip(root, out))
    print(f"[done] outputs in {out}  ({time.perf_counter() - t_run:.1f}s)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Stop as e:
        print(f"[STOP] {e}", file=sys.stderr)
        sys.exit(2)
