#!/usr/bin/env python3
"""
s02b_qp_real_subset.py - classical hinge-loss dual QP on a small TRAINING-ONLY subset of the real data (spec section 5).

100 training rows of the PCA-B representation (50 Low + 50 High, evenly spaced over the training period, standardised with
the frozen train-fitted scaler), labels Low=-1 / High=+1.  The classical hinge-loss dual (linear kernel) is solved with SLSQP
(s02 code) and compared with a MATCHING classical-hinge SVC(kernel='linear').  Two feature widths are used: all 192 features
(likely separable, n < d) and the first 5 components (overlapping classes -> non-trivial slack).
This is separate from Stage A's squared-hinge LinearSVC; no claim of equal optima between the two is made.
Results are written as-is, including any failed check.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import apply_frozen_scaler, load_frozen_scaler, utc_now
import s02_qp_verification as qp


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=".")
    ap.add_argument("--output-dir", default=str(Path(__file__).resolve().parent.parent))
    a = ap.parse_args(argv)
    out, inp = Path(a.output_dir), Path(a.input_dir)
    mean, scale = load_frozen_scaler(out / "models" / "scaler_pca_B.npz")
    with np.load(inp / "features_train_pca_B.npz") as z:
        F, y3, ts = z["F"], z["y"], z["target_timestamp"]
    X = apply_frozen_scaler(F, mean, scale)
    idx = []
    for cls in (0, 2):
        pos = np.flatnonzero(y3 == cls)
        idx.append(pos[np.round(np.linspace(0, len(pos) - 1, 50)).astype(int)])
    idx = np.sort(np.concatenate(idx))
    y = np.where(y3[idx] == 2, 1.0, -1.0)
    res = {"utc": utc_now(), "subset": {"n": int(len(idx)), "n_low": int((y < 0).sum()), "n_high": int((y > 0).sum()),
                                        "first_timestamp": str(ts[idx].min()), "last_timestamp": str(ts[idx].max()), "source": "TRAIN split only"},
           "runs": {}}
    for label, d, C in (("d192_C1", 192, 1.0), ("d5_C1", 5, 1.0), ("d5_C10", 5, 10.0)):
        Xs = X[idx][:, :d]
        passed, diag, alpha, sk = qp.run_qp_test(f"real subset {label}", Xs, y, C, kkt_tol=1e-3, gap_tol=1e-3)
        res["runs"][label] = {"d": d, "C": C, "all_checks_passed": bool(passed), "n_sv": diag["n_support_vectors"], "n_free_sv": diag["n_free_sv"],
                              "n_bound_sv": diag["n_bound_sv"], "bound_violation": diag["bound_violation"],
                              "equality_violation": diag["equality_constraint_violation"], "kkt_stationarity_norm": diag["kkt_stationarity_norm"],
                              "cs1_max": diag["cs1_max_violation"], "cs2_max": diag["cs2_max_violation"], "primal_objective": diag["primal_objective"],
                              "dual_objective": diag["dual_objective"], "primal_dual_gap": diag["primal_dual_gap"],
                              "w_cosine_vs_sklearn": sk["w_cosine_similarity"], "b_qp": diag["b"], "b_sklearn": sk["sklearn_b"],
                              "b_difference": sk["b_difference"], "slack_sum": diag["xi_sum"], "tolerances": {"kkt": 1e-3, "gap": 1e-3, "bound": 1e-8, "equality": 1e-6},
                              "note": "SLSQP (tol 1e-12) vs libsvm SMO (tol 1e-10); bias recovered differently, differences reported not hidden."}
    res["all_runs_passed"] = all(r["all_checks_passed"] for r in res["runs"].values())
    (out / "results" / "qp_real_subset.json").write_text(json.dumps(res, indent=2))
    print("QP_REAL_SUBSET all_passed =", res["all_runs_passed"])


if __name__ == "__main__":
    main()
