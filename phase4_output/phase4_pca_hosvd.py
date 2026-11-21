#!/usr/bin/env python3
"""
Phase 4 -- Matched-dimensional matrix PCA/SVD versus shared-basis truncated HoSVD
for the Beijing Multi-Site Air Quality spatiotemporal tensors (unsupervised only).

Everything is fitted on TRAINING X only.  Common centering tensor
    M[a,b,c] = mean_train_i X[i,a,b,c],   Z_i = X_i - M.
PCA uses the flattened Z (C order, d = 24*12*11 = 3168).  HoSVD uses three shared
orthonormal mode bases (time, station, feature); the sample mode is never compressed.

No SVM / RFF / Markov / classification code lives here.  Labels y are only copied
through to the exported feature files; they are never used by any decomposition.

Usage (from the directory that contains phase4_output/):
  python phase4_output/phase4_pca_hosvd.py --phase2-zip Air_Pollution_Phase2_Processed_Data.zip \
         --output phase4_output --seed 42
  python phase4_output/phase4_pca_hosvd.py --output phase4_output --finalize    # report + ZIP only
"""
import argparse
import csv
import datetime
import hashlib
import json
import logging
import math
import os
import platform
import resource
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import scipy
import scipy.linalg as sla

LOG = logging.getLogger("phase4")
EPS64 = float(np.finfo(np.float64).eps)

SPLITS = ["train", "validation", "test"]
REQUIRED_KEYS = ["X", "y", "target_timestamp", "current_state", "current_pm25"]
EXPECTED_N = {"train": 24108, "validation": 4239, "test": 5720}
EXPECTED_SHAPE = (24, 12, 11)
D_RAW = 24 * 12 * 11
STATIONS = ["Aotizhongxin", "Changping", "Dingling", "Dongsi", "Guanyuan", "Gucheng", "Huairou",
            "Nongzhanguan", "Shunyi", "Tiantan", "Wanliu", "Wanshouxigong"]
MEASURES = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
PHASE2_INFO_FILES = ["training_preprocessing_parameters.npz", "phase2_summary.json", "PHASE2_README.txt"]

# declared configurations: id -> (ranks, exploratory?)
CONFIGS = {
    "A": ((4, 3, 4), False),
    "B": ((8, 4, 6), False),
    "C": ((12, 8, 6), False),
    "B_alt1": ((6, 4, 8), True),   # exploratory k=192 alternate allocation (lower priority)
    "B_alt2": ((8, 6, 4), True),   # exploratory k=192 alternate allocation (lower priority)
}

# ----------------------------------------------------------------------------------------
# tolerances (declared)
# ----------------------------------------------------------------------------------------
ORTHO_ATOL = 1e-8            # max |Q^T Q - I|, rtol = 0, float64 factors
ENERGY_REL_TOL = 1e-9        # relative to ||Z_train||_F^2 : bound / optimality / identity tests
IDENTITY_REL_TOL = 1e-10     # SSE_explicit vs ||Z||^2-||codes||^2, relative to ||Z||^2
CLOSE_GAP_REL = 1e-6         # relative eigengap below which a pair is "close"
SYM_REL_TOL = 1e-10          # Gram asymmetry tolerance (relative to max |G|)


class InputContractError(RuntimeError):
    pass


class NumericalError(RuntimeError):
    pass


# ----------------------------------------------------------------------------------------
# generic helpers
# ----------------------------------------------------------------------------------------
def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def peak_rss_mb():
    """Process high-water-mark resident set size (MB). Cumulative over the whole process."""
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return float("nan")


def now():
    return time.perf_counter()


def setup_logging(out_dir, append=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(out_dir / "execution.log", mode="a" if append else "w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    LOG.addHandler(fh)
    LOG.addHandler(sh)


def environment_info():
    info = {
        "python": sys.version.replace("\n", " "),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
    }
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except Exception:
        info["cpu_model"] = "unavailable"
    try:
        import psutil
        info["psutil"] = psutil.__version__
        info["ram_total_gb"] = round(psutil.virtual_memory().total / 2**30, 3)
        info["ram_available_gb_at_start"] = round(psutil.virtual_memory().available / 2**30, 3)
    except Exception:
        info["ram_total_gb"] = "unavailable"
    try:
        import matplotlib
        info["matplotlib"] = matplotlib.__version__
    except Exception:
        info["matplotlib"] = "unavailable"
    try:
        import pytest
        info["pytest"] = pytest.__version__
    except Exception:
        info["pytest"] = "unavailable"
    try:
        cfg = np.show_config(mode="dicts")
        info["numpy_blas"] = cfg.get("Build Dependencies", {}).get("blas", {})
    except Exception:
        info["numpy_blas"] = "unavailable"
    for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        info[var] = os.environ.get(var, "unset")
    return info


# ----------------------------------------------------------------------------------------
# input contract
# ----------------------------------------------------------------------------------------
def resolve_phase2_inputs(phase2_zip, phase2_dir, tmpdir):
    """Return (paths dict split->Path, info dict).  Stops (InputContractError) when something is missing."""
    paths, info = {}, {"source": None, "sha256": {}, "missing": []}
    needed = {s: f"{s}_windows.npz" for s in SPLITS}
    if phase2_dir:
        base = Path(phase2_dir)
        cand = [base, base / "phase2_output"]
        root = next((c for c in cand if (c / needed["train"]).exists()), None)
        info["source"] = f"directory:{base}"
        if root is None:
            root = base
        for s, fn in needed.items():
            p = root / fn
            if p.exists():
                paths[s] = p
            else:
                info["missing"].append(fn)
        extra_root = root
    elif phase2_zip:
        zp = Path(phase2_zip)
        if not zp.exists():
            raise InputContractError(f"Phase 2 ZIP not found: {zp}")
        info["source"] = f"zip:{zp.name}"
        info["zip_sha256"] = sha256_file(zp)
        extra_root = Path(tmpdir) / "phase2_output"
        extra_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zp) as z:
            names = {Path(n).name: n for n in z.namelist() if not n.endswith("/")}
            info["zip_members"] = sorted(names)
            for fn in list(needed.values()) + PHASE2_INFO_FILES:
                if fn in names:
                    with z.open(names[fn]) as src, open(extra_root / fn, "wb") as dst:
                        shutil.copyfileobj(src, dst)
        for s, fn in needed.items():
            if (extra_root / fn).exists():
                paths[s] = extra_root / fn
            else:
                info["missing"].append(fn)
    else:
        raise InputContractError("Provide --phase2-zip or --phase2-dir.")
    if info["missing"]:
        raise InputContractError("MISSING REQUIRED INPUTS: " + ", ".join(info["missing"]))
    for s, p in paths.items():
        info["sha256"][p.name] = sha256_file(p)
    for fn in PHASE2_INFO_FILES:
        p = Path(extra_root) / fn
        if p.exists():
            info["sha256"][fn] = sha256_file(p)
            info.setdefault("informational_files_present", []).append(fn)
        else:
            info.setdefault("informational_files_absent", []).append(fn)
    return paths, info


def load_split(path):
    with np.load(path, allow_pickle=False) as z:
        keys = list(z.files)
        data = {k: z[k] for k in keys}
    return keys, data


def validate_split(name, keys, d, expected_n=None, expected_shape=EXPECTED_SHAPE):
    """Validate one split; returns report dict or raises InputContractError listing ALL problems."""
    problems = []
    if sorted(keys) != sorted(REQUIRED_KEYS):
        problems.append(f"keys {sorted(keys)} != {sorted(REQUIRED_KEYS)}")
    else:
        X = d["X"]
        n = X.shape[0]
        if X.ndim != 4 or tuple(X.shape[1:]) != tuple(expected_shape):
            problems.append(f"X shape {X.shape}")
        if X.dtype != np.float32:
            problems.append(f"X dtype {X.dtype}")
        if expected_n is not None and n != expected_n:
            problems.append(f"N={n} != expected {expected_n}")
        if not np.isfinite(X).all():
            problems.append("X contains non-finite values")
        for k, dt in [("y", np.int8), ("current_state", np.int8), ("current_pm25", np.float32)]:
            if d[k].dtype != dt:
                problems.append(f"{k} dtype {d[k].dtype} != {np.dtype(dt)}")
            if d[k].shape != (n,):
                problems.append(f"{k} shape {d[k].shape} != ({n},)")
        if d["target_timestamp"].dtype != np.dtype("datetime64[ns]"):
            problems.append(f"target_timestamp dtype {d['target_timestamp'].dtype}")
        if d["target_timestamp"].shape != (n,):
            problems.append("target_timestamp shape mismatch")
        if not problems:
            if not np.isin(d["y"], [0, 1, 2]).all():
                problems.append("y outside {0,1,2}")
            if not np.isin(d["current_state"], [-1, 0, 1, 2]).all():
                problems.append("current_state outside {-1,0,1,2}")
            ts = d["target_timestamp"].astype("int64")
            if n > 1 and not (np.diff(ts) > 0).all():
                problems.append("timestamps not strictly increasing (monotone/duplicates)")
    if problems:
        raise InputContractError(f"[{name}] " + "; ".join(problems))
    cs = d["current_state"]
    pm = d["current_pm25"]
    return {
        "split": name, "N": int(d["X"].shape[0]), "X_shape": list(d["X"].shape),
        "n_current_state_unknown": int((cs == -1).sum()),
        "n_current_pm25_nan": int(np.isnan(pm).sum()),
        "nan_iff_unknown_state": bool(np.array_equal(np.isnan(pm), cs == -1)),
        "class_counts": {int(c): int((d["y"] == c).sum()) for c in (0, 1, 2)},
        "t_first": str(d["target_timestamp"][0]), "t_last": str(d["target_timestamp"][-1]),
        "logical_float32_bytes": int(d["X"].shape[0] * D_RAW * 4),
        "X_min": float(d["X"].min()), "X_max": float(d["X"].max()),
    }


def validate_cross_split(data):
    ts = {s: data[s]["target_timestamp"].astype("int64") for s in data}
    names = [s for s in SPLITS if s in ts]
    problems = []
    for i in range(len(names) - 1):
        a, b = names[i], names[i + 1]
        if ts[a][-1] >= ts[b][0]:
            problems.append(f"chronology violated: last {a} >= first {b}")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if np.intersect1d(ts[names[i]], ts[names[j]]).size:
                problems.append(f"timestamp overlap between {names[i]} and {names[j]}")
    if problems:
        raise InputContractError("; ".join(problems))
    return {"cross_split_chronological": True, "cross_split_overlap": False}


# ----------------------------------------------------------------------------------------
# linear-algebra primitives
# ----------------------------------------------------------------------------------------
def canonicalize_signs(V):
    """Flip each COLUMN so that its largest-|.| entry is non-negative (UPGRADE 2)."""
    V = np.array(V, dtype=np.float64, copy=True)
    piv = np.argmax(np.abs(V), axis=0)
    s = np.sign(V[piv, np.arange(V.shape[1])])
    s[s == 0] = 1.0
    return V * s[None, :]


def canonical_pivot_nonnegative(V):
    piv = np.argmax(np.abs(V), axis=0)
    return bool((V[piv, np.arange(V.shape[1])] >= 0).all())


def orthonormality_error(Q):
    Q = np.asarray(Q, dtype=np.float64)
    return float(np.max(np.abs(Q.T @ Q - np.eye(Q.shape[1]))))


def eig_desc(G, name="gram", use_numpy=False, log=True):
    """Symmetric eigendecomposition, eigenvalues DESCENDING, scale-aware negative-eigenvalue handling.

    tolerance  tol = n * eps64 * max(lambda_max, 0)   (backward-error scale of a dense symmetric eigensolver).
    * eigenvalue < -tol   -> NumericalError (never silently clipped)
    * -tol <= lambda < 0  -> clipped to 0, counted and logged
    Returns dict(w=used eigenvalues, w_raw=raw, V=eigenvectors (columns, sign NOT yet canonical), ...).
    """
    G = np.asarray(G, dtype=np.float64)
    n = G.shape[0]
    scale = float(np.max(np.abs(G))) if G.size else 0.0
    asym = float(np.max(np.abs(G - G.T)) / max(scale, 1e-300))
    if asym > SYM_REL_TOL:
        raise NumericalError(f"{name}: Gram asymmetry {asym:.3e} exceeds {SYM_REL_TOL}")
    Gs = 0.5 * (G + G.T)
    t0 = now()
    if use_numpy:
        w, V = np.linalg.eigh(Gs)
    else:
        w, V = sla.eigh(Gs)
    dt = now() - t0
    w = w[::-1].copy()
    V = np.ascontiguousarray(V[:, ::-1])
    lam_max = max(float(w[0]), 0.0)
    tol = n * EPS64 * lam_max
    min_raw = float(w.min())
    if min_raw < -tol:
        raise NumericalError(f"{name}: significantly negative eigenvalue {min_raw:.3e} < -tol ({-tol:.3e})")
    neg = w < 0
    w_used = np.where(neg, 0.0, w)
    out = {"w": w_used, "w_raw": w, "V": V, "neg_tol": tol, "min_raw_eigenvalue": min_raw,
           "n_negative_clipped": int(neg.sum()), "clipped_abs_sum": float(-w[neg].sum()) if neg.any() else 0.0,
           "n_below_tol": int((w_used <= tol).sum()), "asymmetry_rel": asym, "eigh_seconds": dt,
           "solver": "numpy.linalg.eigh" if use_numpy else "scipy.linalg.eigh"}
    if log:
        LOG.info(f"  eig[{name}] n={n} solver={out['solver']} time={dt:.2f}s lam_max={lam_max:.6e} "
                 f"min_raw={min_raw:.3e} tol={tol:.3e} clipped={out['n_negative_clipped']} "
                 f"(sum |clipped|={out['clipped_abs_sum']:.3e}) n<=tol={out['n_below_tol']} asym={asym:.2e}")
    return out


def tail_energies(w):
    """tail[r] = sum_{j>r} w_j  for r = 0..n  (reverse cumulative sums; no cancellation)."""
    w = np.asarray(w, dtype=np.float64)
    rc = np.concatenate([np.cumsum(w[::-1])[::-1], [0.0]])  # rc[r] = sum_{j>=r}
    return rc  # tail after keeping r = rc[r] (0-based leading r kept)


def projector_distance(Q1, Q2):
    """||Q1 Q1^T - Q2 Q2^T||_F for equal-rank orthonormal Q's, via 2k - 2||Q1^T Q2||_F^2 (no dxd matrix)."""
    k = Q1.shape[1]
    assert Q2.shape[1] == k
    val = 2.0 * k - 2.0 * float(np.sum((Q1.T @ Q2) ** 2))
    return math.sqrt(max(val, 0.0))


def max_principal_angle_deg(Q1, Q2):
    s = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    return float(np.degrees(np.arccos(np.clip(s.min(), -1.0, 1.0))))


def eigengap_diagnostics(w, k, rel=CLOSE_GAP_REL):
    """Relative eigengap criterion: pair (j,j+1) is 'close' if (w_j-w_{j+1}) < rel*w_j."""
    w = np.asarray(w, dtype=np.float64)
    n = len(w)
    kk = min(k, n - 1)
    gaps = (w[:kk] - w[1:kk + 1]) / np.maximum(w[:kk], 1e-300)
    close = gaps < rel
    boundary = float(gaps[k - 1]) if k <= n - 1 else float("nan")
    return {"n_close_pairs_within_retained": int(close[:max(k - 1, 0)].sum()),
            "boundary_pair_close": bool(close[k - 1]) if k <= n - 1 else False,
            "boundary_rel_eigengap": boundary,
            "min_rel_gap_within_first_k": float(gaps[:k].min()) if kk > 0 else float("nan")}


def find_identical_channels(X, atol=0.0):
    """Duplicate (station, feature) series within each measurement: atol=0 -> bitwise identical float32;
    atol>0 -> max |difference| <= atol (e.g. 1e-6, float32 round-off level).
    Returns {feature_name: [[station names of one identical group], ...]} for groups of size >= 2.
    Diagnostic only: explains exact null directions of the flat Gram; nothing is removed or altered."""
    S_, F_ = X.shape[2], X.shape[3]
    out = {}
    for c in range(F_):
        parent = list(range(S_))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for b1 in range(S_):
            for b2 in range(b1 + 1, S_):
                a_, b_ = X[:, :, b1, c], X[:, :, b2, c]
                same = np.array_equal(a_, b_) if atol == 0.0 else bool(np.max(np.abs(a_.astype(np.float64) - b_)) <= atol)
                if same:
                    parent[find(b2)] = find(b1)
        groups = {}
        for b in range(S_):
            groups.setdefault(find(b), []).append(b)
        g = [[STATIONS[b] if S_ == len(STATIONS) else str(b) for b in grp] for grp in groups.values() if len(grp) > 1]
        if g:
            out[MEASURES[c] if F_ == len(MEASURES) else str(c)] = g
    return out


# ----------------------------------------------------------------------------------------
# fitting (training only)
# ----------------------------------------------------------------------------------------
def batches(n, bs):
    for i in range(0, n, bs):
        yield i, min(i + bs, n)


def fit_mean(X, batch):
    t0 = now()
    s = np.zeros(X.shape[1:], dtype=np.float64)
    for i, j in batches(len(X), batch):
        s += X[i:j].astype(np.float64).sum(axis=0)
    M = s / len(X)
    return M, now() - t0


def accumulate_grams(X, M, batch, want_flat=True, want_half_snapshot=True):
    """One streaming pass over TRAIN: flat Gram (d x d), three mode Grams (+ per-batch pieces for split-half),
    ||Z||_F^2.  Timers separate shared (cast+center), flat Gram and mode Gram costs."""
    n = len(X)
    d = int(np.prod(X.shape[1:]))
    T_, S_, F_ = X.shape[1:]
    G = np.zeros((d, d), dtype=np.float64) if want_flat else None
    Gt = np.zeros((T_, T_)); Gs = np.zeros((S_, S_)); Gf = np.zeros((F_, F_))
    per_batch = []
    ssq = 0.0
    tm = {"shared_center": 0.0, "flat_gram": 0.0, "mode_gram": 0.0}
    nb = math.ceil(n / batch)
    half_b = nb // 2 if nb > 1 else 1
    G_half1 = None
    for bi, (i, j) in enumerate(batches(n, batch)):
        t0 = now()
        Z = X[i:j].astype(np.float64) - M
        tm["shared_center"] += now() - t0
        ssq += float(np.vdot(Z, Z))
        if want_flat:
            t0 = now()
            Z2 = Z.reshape(j - i, d)
            G += Z2.T @ Z2
            tm["flat_gram"] += now() - t0
        t0 = now()
        Zt = np.ascontiguousarray(Z.transpose(1, 0, 2, 3)).reshape(T_, -1)
        Zs = np.ascontiguousarray(Z.transpose(2, 0, 1, 3)).reshape(S_, -1)
        Zf = np.ascontiguousarray(Z.transpose(3, 0, 1, 2)).reshape(F_, -1)
        pt, ps, pf = Zt @ Zt.T, Zs @ Zs.T, Zf @ Zf.T
        Gt += pt; Gs += ps; Gf += pf
        per_batch.append((pt, ps, pf))
        tm["mode_gram"] += now() - t0
        if want_flat and want_half_snapshot and bi + 1 == half_b:
            G_half1 = G.copy()
    return {"G": G, "Gt": Gt, "Gs": Gs, "Gf": Gf, "per_batch": per_batch, "ssq": ssq,
            "timers": tm, "G_half1": G_half1, "half_batches": half_b}


def streaming_gram_apply(X, M, batch, Q):
    """Y = A^T (A Q) with A = flatten(X - M), streamed (used only by the truncated fallback)."""
    d = Q.shape[0]
    Y = np.zeros((d, Q.shape[1]), dtype=np.float64)
    for i, j in batches(len(X), batch):
        Z2 = (X[i:j].astype(np.float64) - M).reshape(j - i, d)
        Y += Z2.T @ (Z2 @ Q)
    return Y


def fit_pca_randomized(X, M, k, batch, seed, oversample=20, n_iter=6):
    """APPROXIMATE top-k PCA via streaming randomized subspace iteration on A^T A.
    Returns V (d,k), top-k approximate eigenvalues and convergence diagnostics (relative Ritz residuals)."""
    t0 = now()
    d = int(np.prod(X.shape[1:]))
    rng = np.random.default_rng(seed)
    p = min(k + oversample, d)
    Q, _ = np.linalg.qr(rng.standard_normal((d, p)))
    for _ in range(n_iter):
        Q, _ = np.linalg.qr(streaming_gram_apply(X, M, batch, Q))
    GQ = streaming_gram_apply(X, M, batch, Q)
    B = Q.T @ GQ
    B = 0.5 * (B + B.T)
    th, U = np.linalg.eigh(B)
    th, U = th[::-1], U[:, ::-1]
    V = Q @ U
    GV = GQ @ U                       # = G V exactly (G Q U)
    res = np.linalg.norm(GV - V * th[None, :], axis=0)
    rel = res[:k] / np.maximum(th[:k], 1e-300)
    diag = {"method": "streaming randomized subspace iteration on A^T A (approximate)",
            "oversample": oversample, "n_iter": n_iter, "seed": seed, "block_size": p,
            "max_rel_ritz_residual_top_k": float(rel.max()), "median_rel_ritz_residual_top_k": float(np.median(rel)),
            "seconds": now() - t0}
    return V[:, :k].copy(), th[:k].copy(), diag


# ----------------------------------------------------------------------------------------
# frozen models: forward / inverse transforms
# ----------------------------------------------------------------------------------------
class PCAModel:
    method = "pca"

    def __init__(self, M, V):
        self.M = np.asarray(M, dtype=np.float64)
        self.V = np.asarray(V, dtype=np.float64)
        self.k = self.V.shape[1]
        self.shape = self.M.shape

    def center(self, Xb):
        return Xb.astype(np.float64) - self.M

    def encode_centered(self, Z):
        return Z.reshape(len(Z), -1) @ self.V

    def decode_centered(self, C):
        return (C @ self.V.T).reshape((len(C),) + self.shape)

    def encode(self, Xb):
        return self.encode_centered(self.center(Xb))

    def decode(self, C):
        return self.M + self.decode_centered(np.asarray(C, dtype=np.float64))


class HoSVDModel:
    """Core C_i = Z_i x_time Qt^T x_station Qs^T x_feature Qf^T ; F_i = vec(C_i) in C order (p,q,r)."""
    method = "hosvd"

    def __init__(self, M, Qt, Qs, Qf):
        self.M = np.asarray(M, dtype=np.float64)
        self.Qt, self.Qs, self.Qf = (np.asarray(q, dtype=np.float64) for q in (Qt, Qs, Qf))
        self.ranks = (self.Qt.shape[1], self.Qs.shape[1], self.Qf.shape[1])
        self.k = int(np.prod(self.ranks))
        self.shape = self.M.shape

    def center(self, Xb):
        return Xb.astype(np.float64) - self.M

    def encode_centered(self, Z):
        t = np.einsum("iabc,ap->ipbc", Z, self.Qt, optimize=True)
        t = np.einsum("ipbc,bq->ipqc", t, self.Qs, optimize=True)
        t = np.einsum("ipqc,cr->ipqr", t, self.Qf, optimize=True)
        return t.reshape(len(Z), -1)

    def decode_centered(self, C):
        C = np.asarray(C, dtype=np.float64).reshape((len(C),) + self.ranks)
        t = np.einsum("ipqr,cr->ipqc", C, self.Qf, optimize=True)
        t = np.einsum("ipqc,bq->ipbc", t, self.Qs, optimize=True)
        return np.einsum("ipbc,ap->iabc", t, self.Qt, optimize=True)

    def encode(self, Xb):
        return self.encode_centered(self.center(Xb))

    def decode(self, C):
        return self.M + self.decode_centered(C)

    def kron_basis(self):
        """Independent direct path: (d, k) Kronecker basis whose column (p,q,r) [C order] is
        Qt[:,p] (x) Qs[:,q] (x) Qf[:,r] flattened in C order."""
        return np.kron(np.kron(self.Qt, self.Qs), self.Qf)


def evaluate_split(model, X, batch):
    """Explicit reconstruction on the whole split.  Primary metrics: FULL-PRECISION (float64) internal codes.
    Additional: metrics when decoding from the float32-EXPORTED codes."""
    N = len(X)
    d = int(np.prod(X.shape[1:]))
    F32 = np.empty((N, model.k), dtype=np.float32)
    a = dict(znorm2=0.0, sse=0.0, zhat2=0.0, code2=0.0, sse32=0.0, code2_32=0.0)
    for i, j in batches(N, batch):
        Z = model.center(X[i:j])
        C = model.encode_centered(Z)
        Zh = model.decode_centered(C)
        R = Z - Zh
        a["znorm2"] += float(np.vdot(Z, Z))
        a["sse"] += float(np.vdot(R, R))
        a["zhat2"] += float(np.vdot(Zh, Zh))
        a["code2"] += float(np.vdot(C, C))
        C32 = C.astype(np.float32)
        F32[i:j] = C32
        C32d = C32.astype(np.float64)
        Zh32 = model.decode_centered(C32d)
        R32 = Z - Zh32
        a["sse32"] += float(np.vdot(R32, R32))
        a["code2_32"] += float(np.vdot(C32d, C32d))
    z2 = a["znorm2"]
    out = {
        "N": N, "znorm2": z2, "sse": a["sse"], "zhat2": a["zhat2"],
        "rre": math.sqrt(a["sse"] / z2), "mse": a["sse"] / (N * d), "retained_energy": a["zhat2"] / z2,
        "sse_identity": z2 - a["code2"], "identity_abs_diff": abs(a["sse"] - (z2 - a["code2"])),
        "rre_sq_plus_retained": a["sse"] / z2 + a["zhat2"] / z2,
        "sse_float32_codes": a["sse32"], "rre_float32_codes": math.sqrt(a["sse32"] / z2),
        "mse_float32_codes": a["sse32"] / (N * d), "retained_energy_float32_codes": (z2 - a["sse32"]) / z2,
        "rre_float32_minus_float64": math.sqrt(a["sse32"] / z2) - math.sqrt(a["sse"] / z2),
    }
    return out, F32


def time_op(fn, repeats):
    ts = []
    for _ in range(repeats):
        t0 = now()
        fn()
        ts.append(now() - t0)
    return float(np.median(ts)), float(min(ts)), float(max(ts))


def timed_transform(model, X, batch, repeats):
    def run():
        for i, j in batches(len(X), batch):
            model.encode(X[i:j]).astype(np.float32)
    return time_op(run, repeats)


def timed_inverse(model, F32, batch, repeats):
    def run():
        for i, j in batches(len(F32), batch):
            model.decode(F32[i:j].astype(np.float64)).astype(np.float32)
    return time_op(run, repeats)


def subset_direct_check(model, X, idx):
    """Independent direct path on a subset: explicit projector / Kronecker basis (no einsum, no shortcuts).
    Returns max |difference| between direct and batched pipeline reconstructions and the SSE shortcut gap."""
    Xs = X[idx]
    Z = model.center(Xs)
    d = Z[0].size
    A = Z.reshape(len(Z), d)
    B = model.V if model.method == "pca" else model.kron_basis()
    P_codes = A @ B
    Zh_direct = (P_codes @ B.T).reshape(Z.shape)
    Zh_pipe = model.decode_centered(model.encode_centered(Z))
    sse_direct = float(np.sum((Z - Zh_direct) ** 2))
    shortcut = float(np.sum(Z ** 2) - np.sum(P_codes ** 2))
    codes_pipe = model.encode_centered(Z)
    return {"max_abs_recon_diff": float(np.max(np.abs(Zh_direct - Zh_pipe))),
            "max_abs_code_diff": float(np.max(np.abs(P_codes - codes_pipe))),
            "sse_direct": sse_direct, "sse_shortcut": shortcut,
            "shortcut_rel_gap": abs(sse_direct - shortcut) / max(float(np.sum(Z ** 2)), 1e-300)}


# ----------------------------------------------------------------------------------------
# storage accounting
# ----------------------------------------------------------------------------------------
def storage_numbers(method, k, ranks, N, d, model_dtype_bytes=8, code_bytes=4, mode_dims=(24, 12, 11)):
    mean_bytes = d * model_dtype_bytes
    if method == "pca":
        basis_params = d * k
    else:
        basis_params = mode_dims[0] * ranks[0] + mode_dims[1] * ranks[1] + mode_dims[2] * ranks[2]
    basis_bytes = basis_params * model_dtype_bytes
    model_bytes = mean_bytes + basis_bytes
    feat_bytes = N * k * code_bytes
    raw = N * d * 4
    total = feat_bytes + model_bytes
    return {"raw_logical_bytes": raw, "model_parameter_count": d + basis_params, "mean_bytes": mean_bytes,
            "basis_bytes": basis_bytes, "model_bytes": model_bytes, "feature_bytes": feat_bytes,
            "total_estimated_bytes": total, "dimension_ratio": d / k, "effective_storage_ratio": raw / total}


# ----------------------------------------------------------------------------------------
# config planning (resource-aware, transparent)
# ----------------------------------------------------------------------------------------
def estimate_exact_gram_memory_gb(n_train, d):
    x_train = n_train * d * 4
    g, v, g_half, alt_v = [d * d * 8] * 4
    workspace = 3 * d * d * 8
    batch_buf = 1024 * d * 8 * 4
    return (x_train + g + v + g_half + alt_v + workspace + batch_buf) / 2**30


def choose_pca_solver(requested, n_train, d, max_memory_gb, available_gb):
    """Returns (solver, reason).  'exact-gram' and 'truncated-fallback' are never silently swapped."""
    need = estimate_exact_gram_memory_gb(n_train, d)
    budget = max_memory_gb if max_memory_gb else (available_gb * 0.8 if available_gb else None)
    if requested == "exact-gram":
        return "exact-gram", f"forced by user; estimated peak {need:.2f} GB (budget {budget})"
    if requested == "truncated-fallback":
        return "truncated-fallback", "forced by user"
    if budget is None or need <= budget:
        return "exact-gram", f"auto: estimated peak {need:.2f} GB <= budget {budget if budget else 'unknown'} GB"
    return "truncated-fallback", f"auto: estimated exact-Gram peak {need:.2f} GB > budget {budget:.2f} GB"


def plan_configs(requested_ids, solver, fallback_max_k):
    """Return list of dicts config_id, ranks, k, run(bool), status, reason."""
    plan = []
    for cid in requested_ids:
        if cid not in CONFIGS:
            raise ValueError(f"unknown config id {cid}; known: {list(CONFIGS)}")
        r, expl = CONFIGS[cid]
        k = int(np.prod(r))
        run, status, reason = True, "PLANNED", ""
        if solver == "truncated-fallback" and k > fallback_max_k:
            run, status = False, "NOT RUN"
            reason = f"approximate fallback solver restricted to k<={fallback_max_k}; k={k}"
        plan.append({"config_id": cid, "ranks": r, "k": k, "exploratory": expl, "run": run,
                     "status": status, "reason": reason})
    return plan


# ----------------------------------------------------------------------------------------
# toy / hand-worked examples (ILLUSTRATIVE ONLY -- not Beijing findings)
# ----------------------------------------------------------------------------------------
def toy_pca_2x2():
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    G = A.T @ A
    w, V = np.linalg.eigh(G)
    w, V = w[::-1], V[:, ::-1]
    V = canonicalize_signs(V)
    v1 = V[:, :1]
    Ahat = A @ v1 @ v1.T
    return {"A": A, "gram": G, "eigenvalues": w, "v1": v1.ravel(), "rank1_recon": Ahat,
            "sse": float(np.sum((A - Ahat) ** 2)), "retained_fraction": float(w[0] / w.sum())}


def full_hosvd_3way(T, ranks):
    """Classical truncated HOSVD of a 3-way tensor (all three modes compressed) -- toy use only."""
    Qs, grams, evals = [], [], []
    for mode in range(3):
        U = np.moveaxis(T, mode, 0).reshape(T.shape[mode], -1)
        G = U @ U.T
        w, V = np.linalg.eigh(G)
        w, V = w[::-1], canonicalize_signs(V[:, ::-1])
        grams.append(G); evals.append(w); Qs.append(V[:, :ranks[mode]])
    core = np.einsum("abc,ap,bq,cr->pqr", T, *Qs)
    That = np.einsum("pqr,ap,bq,cr->abc", core, *Qs)
    tails = [float(e[r:].sum()) for e, r in zip(evals, ranks)]
    return {"grams": grams, "eigenvalues": evals, "Q": Qs, "core": core, "That": That,
            "sse": float(np.sum((T - That) ** 2)), "tails": tails, "bound": float(sum(tails))}


def toy_hosvd_2x2x2():
    T = np.zeros((2, 2, 2))
    T[0, 0, 0] = 2.0
    T[1, 1, 1] = 1.0
    return full_hosvd_3way(T, (1, 1, 1))


# ----------------------------------------------------------------------------------------
# I/O of outputs
# ----------------------------------------------------------------------------------------
def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        for r in rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(float(v)) if isinstance(v, (float, np.floating)) and math.isfinite(float(v))
                            else ("" if v is None else v)) for k, v in r.items()})


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_features(path, F, meta):
    np.savez_compressed(path, F=F, y=meta["y"], target_timestamp=meta["target_timestamp"],
                        current_state=meta["current_state"], current_pm25=meta["current_pm25"])


def verify_feature_file(path, F, meta, n_expected, k_expected):
    """Reload the saved file; verify exact key set, dtypes, shapes and bit-exact metadata (NaN-aware pm25)."""
    with np.load(path, allow_pickle=False) as z:
        keys = list(z.files)
        L = {k: z[k] for k in keys}
    ok = keys == ["F", "y", "target_timestamp", "current_state", "current_pm25"]
    ok &= L["F"].dtype == np.float32 and L["F"].shape == (n_expected, k_expected)
    ok &= L["y"].dtype == np.int8 and L["current_state"].dtype == np.int8
    ok &= L["target_timestamp"].dtype == np.dtype("datetime64[ns]") and L["current_pm25"].dtype == np.float32
    ok &= np.array_equal(L["F"], F)
    ok &= np.array_equal(L["y"], meta["y"]) and np.array_equal(L["current_state"], meta["current_state"])
    ok &= np.array_equal(L["target_timestamp"], meta["target_timestamp"])
    ok &= np.array_equal(L["current_pm25"], meta["current_pm25"], equal_nan=True)
    ok &= L["current_pm25"].tobytes() == meta["current_pm25"].tobytes()   # also bit-exact incl. NaN payloads
    ok &= bool(np.isfinite(L["F"]).all())
    return bool(ok)


# ----------------------------------------------------------------------------------------
# main experiment
# ----------------------------------------------------------------------------------------
def run_experiment(args):
    t_start = now()
    out = Path(args.output).resolve()
    setup_logging(out, append=False)
    (out / "figures").mkdir(exist_ok=True)
    env = environment_info()
    script_path = Path(__file__).resolve()
    meta = {"phase": 4, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "command_line": " ".join(sys.argv), "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "environment": env, "seed": args.seed, "code_sha256": sha256_file(script_path),
            "status": "RUNNING", "tolerances": {
                "orthonormality_atol": ORTHO_ATOL, "energy_rel_tol_vs_train_total_energy": ENERGY_REL_TOL,
                "sse_identity_rel_tol": IDENTITY_REL_TOL, "close_eigengap_rel": CLOSE_GAP_REL,
                "gram_symmetry_rel_tol": SYM_REL_TOL,
                "negative_eigenvalue_tol": "n * eps64 * lambda_max (see eig_desc)"}}
    LOG.info("=" * 100)
    LOG.info("PHASE 4: PCA/SVD vs shared-basis truncated HoSVD  (unsupervised reconstruction analysis)")
    LOG.info(f"command: {meta['command_line']}")
    LOG.info(f"environment: {json.dumps({k: env[k] for k in ['python', 'numpy', 'scipy', 'platform', 'cpu_model', 'cpu_count_logical', 'ram_total_gb'] if k in env})}")
    LOG.info(f"script sha256: {meta['code_sha256']}")

    tmp = tempfile.mkdtemp(prefix="phase4_in_")
    try:
        # ---------------- 1. input contract ----------------
        try:
            paths, in_info = resolve_phase2_inputs(args.phase2_zip, args.phase2_dir, tmp)
        except InputContractError as e:
            LOG.info(f"STOP: {e}")
            meta["status"] = f"STOPPED: {e}"
            json.dump(meta, open(out / "phase4_metadata.json", "w"), indent=2)
            raise
        meta["inputs"] = in_info
        LOG.info(f"inputs from {in_info['source']}; sha256:")
        for k, v in in_info["sha256"].items():
            LOG.info(f"   {k}: {v}")
        data, reports = {}, {}
        for s in SPLITS:
            keys, d = load_split(paths[s])
            reports[s] = validate_split(s, keys, d, None if args.skip_count_check else EXPECTED_N[s])
            data[s] = d
            LOG.info(f"[contract] {s}: N={reports[s]['N']} X{tuple(reports[s]['X_shape'])} float32 finite OK; "
                     f"current_state==-1: {reports[s]['n_current_state_unknown']}; NaN current_pm25: "
                     f"{reports[s]['n_current_pm25_nan']} (NaN iff state==-1: {reports[s]['nan_iff_unknown_state']}); "
                     f"classes {reports[s]['class_counts']}; {reports[s]['t_first']} .. {reports[s]['t_last']}; "
                     f"raw logical bytes {reports[s]['logical_float32_bytes']}")
        meta["cross_split"] = validate_cross_split(data)
        meta["input_validation"] = reports
        LOG.info(f"[contract] cross-split: {meta['cross_split']}")
        Xtr = data["train"]["X"]
        n_train = len(Xtr)
        d = int(np.prod(Xtr.shape[1:]))
        for s, expect in [("train", 305496576), ("validation", 53716608), ("test", 72483840)]:
            if not args.skip_count_check:
                assert reports[s]["logical_float32_bytes"] == expect, f"raw bytes mismatch {s}"
        dup = find_identical_channels(Xtr)                 # bitwise identical
        dup_near = find_identical_channels(Xtr, atol=1e-6)  # identical up to float32 round-off
        n_exact = sum(len(g) - 1 for groups in dup.values() for g in groups)
        n_redundant = sum(len(g) - 1 for groups in dup_near.values() for g in groups)
        meta["identical_channel_groups_train_bitwise"] = dup
        meta["identical_channel_groups_train_within_1e-6"] = dup_near
        meta["n_redundant_channels_bitwise"] = n_exact
        meta["n_redundant_station_feature_channels"] = n_redundant
        meta["implied_exact_null_directions_of_flat_gram"] = n_redundant * Xtr.shape[1]
        LOG.info(f"[data-diagnostic] TRAIN station-feature series identical to within 1e-6 (float32 round-off): {n_redundant} redundant channels "
                 f"({n_exact} bitwise identical) -> {n_redundant * Xtr.shape[1]} (null) directions expected in the flat Gram; groups(<=1e-6): {dup_near}")
        meta["flatten_order"] = "C (time, station, feature) -> index = a*(12*11) + b*11 + c"
        meta["station_order"] = STATIONS
        meta["measurement_order"] = MEASURES

        # ---------------- 2. solver feasibility ----------------
        avail = env.get("ram_available_gb_at_start")
        avail = avail if isinstance(avail, (int, float)) else None
        solver, reason = choose_pca_solver(args.pca_solver, n_train, d, args.max_memory_gb, avail)
        LOG.info(f"[solver] PCA solver = {solver} ({reason}). Dense symmetric eigensolve is O(d^3) "
                 f"(d={d}); stored Gram = {d * d * 8} bytes ({d * d * 8 / 1e6:.2f} MB); workspace is solver dependent.")
        requested = [c.strip() for c in args.configs.split(",") if c.strip()]
        plan = plan_configs(requested, solver, args.fallback_max_k)
        meta["pca_solver"] = {"requested": args.pca_solver, "used": solver, "reason": reason}

        # ---------------- 3. fit (TRAIN only) ----------------
        LOG.info("[fit] centering tensor M from TRAIN only (float64)")
        M, t_mean = fit_mean(Xtr, args.batch_size)
        LOG.info(f"  M: shape {M.shape}, mean fit {t_mean:.2f}s, |M|max={np.abs(M).max():.4f}")
        want_flat = solver == "exact-gram"
        LOG.info(f"[fit] streaming Gram accumulation (batch={args.batch_size}, flat Gram={'yes' if want_flat else 'no (fallback)'})")
        acc = accumulate_grams(Xtr, M, args.batch_size, want_flat=want_flat)
        tm = acc["timers"]
        LOG.info(f"  timers: shared_center={tm['shared_center']:.2f}s flat_gram={tm['flat_gram']:.2f}s mode_gram={tm['mode_gram']:.2f}s; ||Z_train||_F^2={acc['ssq']:.10e}")
        train_energy = acc["ssq"]
        rss_after_gram = peak_rss_mb()

        # -- HoSVD mode eigendecompositions
        mode_names = ["time", "station", "feature"]
        mode_grams = [acc["Gt"], acc["Gs"], acc["Gf"]]
        mode_eig, t_mode_eig = {}, 0.0
        for nm, G in zip(mode_names, mode_grams):
            e = eig_desc(G, f"mode-{nm}")
            e["V"] = canonicalize_signs(e["V"])
            assert canonical_pivot_nonnegative(e["V"])
            mode_eig[nm] = e
            t_mode_eig += e["eigh_seconds"]
            tr_rel = abs(float(np.trace(G)) - train_energy) / train_energy
            LOG.info(f"  mode-{nm}: trace(G)={np.trace(G):.10e} vs ||Z||^2 rel diff {tr_rel:.2e}; orthonormality err {orthonormality_error(e['V']):.2e}")
            meta.setdefault("mode_eig_diag", {})[nm] = {k: e[k] for k in
                ["neg_tol", "min_raw_eigenvalue", "n_negative_clipped", "clipped_abs_sum", "n_below_tol", "asymmetry_rel", "eigh_seconds", "solver"]}
            meta["mode_eig_diag"][nm]["trace_rel_diff_vs_train_energy"] = tr_rel

        # -- PCA
        pca_info = {"solver": solver, "exact": solver == "exact-gram"}
        stab_rows = []
        V_full = None
        w_pca = None
        if solver == "exact-gram":
            G = acc["G"]
            asym = float(np.max(np.abs(G - G.T)) / np.max(np.abs(G)))
            LOG.info(f"[fit] exact flat Gram: shape {G.shape}, asymmetry rel {asym:.2e}, trace {np.trace(G):.10e}")
            # consistency of mode Grams with partial traces of the flat Gram (independent accumulation path)
            G6 = G.reshape(24, 12, 11, 24, 12, 11)
            for nm, ref in zip(mode_names, [np.einsum("abcdbc->ad", G6), np.einsum("abcaec->be", G6), np.einsum("abcabf->cf", G6)]):
                cmp = {"time": acc["Gt"], "station": acc["Gs"], "feature": acc["Gf"]}[nm]
                rel = float(np.max(np.abs(ref - cmp)) / np.max(np.abs(cmp)))
                LOG.info(f"  mode-{nm} Gram vs partial trace of flat Gram: max rel diff {rel:.2e}")
                meta["mode_eig_diag"][nm]["flat_gram_partial_trace_rel_diff"] = rel
            del G6
            ep = eig_desc(G, "pca-flat-gram (scipy)")
            V_full = canonicalize_signs(ep["V"])
            assert canonical_pivot_nonnegative(V_full)
            w_pca = ep["w"]
            pca_info.update({k: ep[k] for k in ["neg_tol", "min_raw_eigenvalue", "n_negative_clipped", "clipped_abs_sum",
                                                 "n_below_tol", "asymmetry_rel", "eigh_seconds"]})
            pca_info["trace_rel_diff_vs_train_energy"] = abs(float(w_pca.sum()) - train_energy) / train_energy
            pca_info["eigenvalue_sum_over_trace_gram_rel_diff"] = abs(float(ep["w_raw"].sum()) - float(np.trace(G))) / float(np.trace(G))
            pca_info["numerical_rank_above_tol"] = int((w_pca > ep["neg_tol"]).sum())
            LOG.info(f"  sum eig vs ||Z||^2 rel diff {pca_info['trace_rel_diff_vs_train_energy']:.2e}; numerical rank (> tol) {pca_info['numerical_rank_above_tol']} of {d}")
            # alt solver for stability diagnostic (same Gram, different LAPACK driver path)
            ep_alt = eig_desc(G, "pca-flat-gram (numpy alt)", use_numpy=True)
            V_alt = canonicalize_signs(ep_alt["V"])
            # split-half sensitivity (chronological halves of TRAIN)
            G1 = acc["G_half1"]
            G2 = G - G1
            e1 = eig_desc(G1, "pca-half1", log=False); e2 = eig_desc(G2, "pca-half2", log=False)
            V1, V2 = canonicalize_signs(e1["V"]), canonicalize_signs(e2["V"])
            del G1, G2
            pca_stab_inputs = {"V_alt": V_alt, "V1": V1, "V2": V2}
            LOG.info(f"  split-half: first {acc['half_batches']} of {len(acc['per_batch'])} train batches vs remainder")
            acc["G"] = None
            del G
        else:
            kmax = max(c["k"] for c in plan if c["run"])
            LOG.info(f"[fit] APPROXIMATE truncated fallback for k_max={kmax} (spectrum will be INCOMPLETE)")
            Vk_fb, th_fb, diag_fb = fit_pca_randomized(Xtr, M, kmax, args.batch_size, args.seed)
            V_full = canonicalize_signs(Vk_fb)
            w_pca = th_fb
            pca_info["fallback_diag"] = diag_fb
            pca_stab_inputs = None
            LOG.info(f"  fallback diag: {diag_fb}")

        t_pca_eig = pca_info.get("eigh_seconds", pca_info.get("fallback_diag", {}).get("seconds", 0.0))
        fit_seconds = {
            "pca": t_mean + tm["shared_center"] + tm["flat_gram"] + t_pca_eig if solver == "exact-gram" else t_mean + pca_info["fallback_diag"]["seconds"],
            "hosvd": t_mean + tm["shared_center"] + tm["mode_gram"] + t_mode_eig,
        }
        meta["fit_seconds"] = {**fit_seconds, "components": {"mean_pass": t_mean, **tm, "pca_eig": t_pca_eig, "mode_eig_total": t_mode_eig},
                               "note": "fit_seconds are one-time, shared by all configurations of that method; includes mean pass, "
                                       "batch cast+centering, the method's own Gram accumulation and its eigensolve(s). Single-core wall clock; "
                                       "the two methods were accumulated in the same data pass, with per-operation timers."}
        LOG.info(f"[fit] fit_seconds PCA={fit_seconds['pca']:.2f}  HoSVD={fit_seconds['hosvd']:.2f}")
        fit_peak = peak_rss_mb()

        # ---------------- 4. spectra files ----------------
        total_e = train_energy
        if solver == "exact-gram":
            wtail = tail_energies(w_pca)
            cum = np.cumsum(w_pca)
            rows, marks = [], {}
            for cid, (r, _) in CONFIGS.items():
                marks.setdefault(int(np.prod(r)), []).append(cid)
            for j in range(1, d + 1):
                rows.append({"component": j, "eigenvalue": float(w_pca[j - 1]), "eigenvalue_raw": float(ep["w_raw"][j - 1]),
                             "singular_value": math.sqrt(max(w_pca[j - 1], 0.0)),
                             "retained_energy_fraction_if_k_equals_j": float(cum[j - 1] / total_e),
                             "tail_energy_fraction_after_j": float(wtail[j] / total_e), "tail_energy_abs_after_j": float(wtail[j]),
                             "total_energy_train": total_e, "numerically_zero": bool(w_pca[j - 1] <= ep["neg_tol"]),
                             "config_ids_with_this_k": ";".join(marks.get(j, [])), "status": "complete_exact_gram_spectrum"})
            write_csv(out / "pca_components_spectrum.csv", rows)
        else:
            rows = [{"component": j + 1, "eigenvalue": float(w_pca[j]), "singular_value": math.sqrt(max(w_pca[j], 0.0)),
                     "status": "APPROXIMATE_INCOMPLETE_spectrum_only_top_Ritz_values", "tail_energy_fraction_after_j": "NA",
                     "total_energy_train": total_e} for j in range(len(w_pca))]
            write_csv(out / "pca_components_spectrum.csv", rows)
        mrows = []
        for nm in mode_names:
            w = mode_eig[nm]["w"]
            tl = tail_energies(w)
            cm = np.cumsum(w)
            for j in range(1, len(w) + 1):
                mrows.append({"mode": nm, "index": j, "eigenvalue": float(w[j - 1]), "singular_value": math.sqrt(max(w[j - 1], 0.0)),
                              "retained_energy_fraction": float(cm[j - 1] / total_e), "tail_energy_abs_after_j": float(tl[j]),
                              "tail_energy_fraction_after_j": float(tl[j] / total_e), "total_energy_train": total_e,
                              "ranks_selected_by": ";".join(f"{cid}" for cid, (r, _) in CONFIGS.items()
                                                            if r[mode_names.index(nm)] == j)})
        write_csv(out / "hosvd_mode_spectra.csv", mrows)

        # ---------------- 5. per-config models ----------------
        completed = []
        comp_rows, stor_rows, bound_rows, manifest_rows, kenergy_rows = [], [], [], [], []
        direct_checks, status_rows = {}, []
        Qfull = {nm: mode_eig[nm]["V"] for nm in mode_names}
        rng = np.random.default_rng(args.seed)
        subset_idx = {s: np.sort(rng.choice(len(data[s]["X"]), size=min(64, len(data[s]["X"])), replace=False)) for s in SPLITS}
        fitted_arrays = {"M": M, "Q_time_full": Qfull["time"], "Q_station_full": Qfull["station"], "Q_feature_full": Qfull["feature"],
                         "eigvals_time": mode_eig["time"]["w"], "eigvals_station": mode_eig["station"]["w"],
                         "eigvals_feature": mode_eig["feature"]["w"], "train_energy": np.array(total_e),
                         "flatten_order": np.array("C"), "model_dtype": np.array("float64")}
        pca_models, hosvd_models, split_results = {}, {}, {}

        for pc in plan:
            cid, r, k = pc["config_id"], pc["ranks"], pc["k"]
            if not pc["run"]:
                LOG.info(f"[config {cid}] NOT RUN: {pc['reason']}")
                status_rows.append({"config_id": cid, "k": k, "status": "NOT RUN", "reason": pc["reason"]})
                continue
            LOG.info(f"[config {cid}] ranks={r} k={k} d/k={d / k:.2f}" + ("  (EXPLORATORY additional experiment)" if pc["exploratory"] else ""))
            # HoSVD factors for this config
            Qt, Qs, Qf = (Qfull[nm][:, :rr].copy() for nm, rr in zip(mode_names, r))
            for nm, Q in zip(mode_names, (Qt, Qs, Qf)):
                err = orthonormality_error(Q)
                assert err <= ORTHO_ATOL, f"Q_{nm} orthonormality {err}"
                assert canonical_pivot_nonnegative(Q)
            hmodel = HoSVDModel(M, Qt, Qs, Qf)
            hosvd_models[cid] = hmodel
            fitted_arrays[f"{cid}_ranks"] = np.array(r)
            fitted_arrays[f"{cid}_Q_time"], fitted_arrays[f"{cid}_Q_station"], fitted_arrays[f"{cid}_Q_feature"] = Qt, Qs, Qf
            run_pca = not pc["exploratory"]
            if run_pca:
                Vk = V_full[:, :k].copy()
                oerr = orthonormality_error(Vk)
                assert oerr <= ORTHO_ATOL, f"PCA V_k orthonormality {oerr}"
                assert canonical_pivot_nonnegative(Vk)
                pmodel = PCAModel(M, Vk)
                pca_models[cid] = pmodel
                LOG.info(f"  PCA V_k orthonormality err {oerr:.2e}; HoSVD factor errs " + ", ".join(f"{orthonormality_error(q):.2e}" for q in (Qt, Qs, Qf)))
                np.savez(out / f"fitted_pca_basis_{k}.npz", V=Vk, M=M, eigenvalues_top_k=w_pca[:k],
                         solver=np.array(solver), is_exact=np.array(solver == "exact-gram"), flatten_order=np.array("C"),
                         model_dtype=np.array("float64"), canonical_sign_pivot_nonneg=np.array(True))
                # PCA stability diagnostics
                if pca_stab_inputs is not None:
                    gap = eigengap_diagnostics(w_pca, k)
                    for comp, Vb in [("scipy_vs_numpy_eigh_same_gram", pca_stab_inputs["V_alt"][:, :k]),
                                     ("split_half_train_first_vs_second", None)]:
                        if Vb is None:
                            Va, Vc = pca_stab_inputs["V1"][:, :k], pca_stab_inputs["V2"][:, :k]
                        else:
                            Va, Vc = Vk, Vb
                        stab_rows.append({"object": "pca", "config_id": cid, "dimension": k, "comparison": comp,
                                          "projector_frobenius_distance": projector_distance(Va, Vc),
                                          "max_principal_angle_deg": max_principal_angle_deg(Va, Vc), **gap})
                kenergy_rows.append({"method": "pca", "config_id": cid, "k": k, "retained_energy_fraction": float(np.sum(w_pca[:k]) / total_e) if solver == "exact-gram" else "NA",
                                     "tail_energy_fraction": float(wtail[k] / total_e) if solver == "exact-gram" else "NA",
                                     "tail_energy_abs": float(wtail[k]) if solver == "exact-gram" else "NA", "total_energy_train": total_e})
            # HoSVD stability diagnostics
            for mi, nm in enumerate(mode_names):
                rr = r[mi]
                w_full = mode_eig[nm]["w"]
                gap = eigengap_diagnostics(w_full, rr)
                alt = eig_desc(mode_grams[mi], f"alt-{nm}", use_numpy=True, log=False)
                Valt = canonicalize_signs(alt["V"])[:, :rr]
                half = len(acc["per_batch"]) // 2 if len(acc["per_batch"]) > 1 else 1
                Gh1 = sum(pb[mi] for pb in acc["per_batch"][:half]); Gh2 = sum(pb[mi] for pb in acc["per_batch"][half:])
                Vh1 = canonicalize_signs(eig_desc(Gh1, "h1", log=False)["V"])[:, :rr]
                Vh2 = canonicalize_signs(eig_desc(Gh2, "h2", log=False)["V"])[:, :rr]
                Qm = (Qt, Qs, Qf)[mi]
                for comp, (Va, Vc) in [("scipy_vs_numpy_eigh_same_gram", (Qm, Valt)), ("split_half_train_first_vs_second", (Vh1, Vh2))]:
                    stab_rows.append({"object": f"hosvd_{nm}", "config_id": cid, "dimension": rr, "comparison": comp,
                                      "projector_frobenius_distance": projector_distance(Va, Vc),
                                      "max_principal_angle_deg": max_principal_angle_deg(Va, Vc), **gap})
            # bases for tail energies
            tails = [float(tail_energies(mode_eig[nm]["w"])[rr]) for nm, rr in zip(mode_names, r)]
            kenergy_rows.append({"method": "hosvd", "config_id": cid, "k": k, "retained_energy_fraction": "NA(not additive)",
                                 "tail_energy_abs": "time/station/feature=" + "/".join(f"{t:.6e}" for t in tails),
                                 "tail_energy_fraction": sum(tails) / total_e, "total_energy_train": total_e})

            # -------- evaluate each split / method --------
            methods = [("hosvd", hmodel)] + ([("pca", pmodel)] if run_pca else [])
            for method, model in methods:
                for s in SPLITS:
                    X = data[s]["X"]
                    t_eval = now()
                    res, F32 = evaluate_split(model, X, args.batch_size)
                    t_eval = now() - t_eval
                    tr_med, tr_min, tr_max = timed_transform(model, X, args.batch_size, args.timing_repeats)
                    inv_med, inv_min, inv_max = timed_inverse(model, F32, args.batch_size, args.timing_repeats)
                    chk = subset_direct_check(model, X, subset_idx[s])
                    direct_checks[f"{method}_{cid}_{s}"] = chk
                    assert chk["max_abs_recon_diff"] < 1e-9, f"direct-path mismatch {method} {cid} {s}: {chk}"
                    # features + metadata
                    mt = {k_: data[s][k_] for k_ in ["y", "target_timestamp", "current_state", "current_pm25"]}
                    fpath = out / f"features_{s}_{method}_{cid}.npz"
                    save_features(fpath, F32, mt)
                    aligned = verify_feature_file(fpath, F32, mt, len(X), model.k)
                    assert aligned, f"feature file verification failed {fpath}"
                    sha = sha256_file(fpath)
                    stor = storage_numbers(method, k, r, len(X), d)
                    fname_k = {"hosvd": r, "pca": (None, None, None)}[method]
                    peak = peak_rss_mb()
                    exact_flag = (method == "hosvd") or (solver == "exact-gram")
                    row = {"split": s, "method": method, "config_id": cid, "k": k,
                           "r_time": fname_k[0], "r_station": fname_k[1], "r_feature": fname_k[2],
                           "relative_reconstruction_error": res["rre"], "mse": res["mse"], "retained_energy": res["retained_energy"],
                           "fit_seconds": fit_seconds[method], "transform_seconds": tr_med, "inverse_seconds": inv_med,
                           "model_bytes": stor["model_bytes"], "feature_bytes": stor["feature_bytes"],
                           "total_estimated_bytes": stor["total_estimated_bytes"], "peak_ram_mb": peak,
                           "sse": res["sse"], "znorm2": res["znorm2"], "zhat_norm2": res["zhat2"], "n_samples": len(X),
                           "sse_identity_znorm2_minus_codes2": res["sse_identity"], "identity_abs_diff": res["identity_abs_diff"],
                           "rre_sq_plus_retained": res["rre_sq_plus_retained"],
                           "sse_float32_codes": res["sse_float32_codes"], "rre_float32_codes": res["rre_float32_codes"],
                           "mse_float32_codes": res["mse_float32_codes"], "rre_float32_minus_float64": res["rre_float32_minus_float64"],
                           "transform_seconds_min": tr_min, "transform_seconds_max": tr_max, "inverse_seconds_min": inv_min,
                           "inverse_seconds_max": inv_max, "timing_repeats": args.timing_repeats, "evaluate_pass_seconds": t_eval,
                           "fit_shared_across_configs": True, "solver": ("scipy.linalg.eigh(Gram)" if method == "pca" and solver == "exact-gram"
                                                                         else ("approximate_randomized_subspace_iteration" if method == "pca" else "scipy.linalg.eigh(mode Grams)")),
                           "pca_exact": (solver == "exact-gram") if method == "pca" else "", "exploratory_config": pc["exploratory"],
                           "metric_basis": "float64 internal codes; *_float32_codes columns use exported float32 codes",
                           "unit": "Phase-2 standardized units (not ug/m^3)", "status": "completed"}
                    comp_rows.append(row)
                    stor_rows.append({"split": s, "method": method, "config_id": cid, "k": k, **stor,
                                      "model_dtype": "float64", "feature_dtype": "float32", "code_bytes_per_scalar": 4,
                                      "model_bytes_per_scalar": 8, "actual_feature_file_bytes": os.path.getsize(fpath),
                                      "exploratory_config": pc["exploratory"]})
                    manifest_rows.append({"split": s, "method": method, "config_id": cid, "k": k,
                                          "r_time": fname_k[0], "r_station": fname_k[1], "r_feature": fname_k[2],
                                          "relative_path": fpath.name, "sha256": sha, "n_samples": len(X), "n_features": model.k,
                                          "metadata_alignment_passed": aligned, "actual_file_bytes": os.path.getsize(fpath),
                                          "sha256_note": "hash stable within a fixed software environment only; not claimed cross-platform"})
                    split_results[(method, cid, s)] = res
                    LOG.info(f"  {method:5s} {cid:6s} {s:10s} RRE={res['rre']:.6f} MSE={res['mse']:.6f} ret={res['retained_energy']:.6f} "
                             f"(1-RRE^2-ret={1 - res['rre_sq_plus_retained']:.1e}) fp32-code dRRE={res['rre_float32_minus_float64']:.1e} "
                             f"transform={tr_med:.3f}s inverse={inv_med:.3f}s peakRSS={peak:.0f}MB")
                    del F32
            # bound + optimality check on TRAIN
            h_tr = split_results[("hosvd", cid, "train")]
            tol = ENERGY_REL_TOL * total_e
            bound = sum(tails)
            passed = bool(h_tr["sse"] <= bound + tol)
            bound_rows.append({"config_id": cid, "r_time": r[0], "r_station": r[1], "r_feature": r[2], "k": k,
                               "empirical_train_sse": h_tr["sse"], "tail_time": tails[0], "tail_station": tails[1], "tail_feature": tails[2],
                               "upper_bound_sse": bound, "numerical_tolerance": tol, "bound_passed": passed,
                               "margin": bound - h_tr["sse"], "relative_slack": (bound - h_tr["sse"]) / bound if bound > 0 else "",
                               "sse_over_bound": h_tr["sse"] / bound if bound > 0 else "",
                               "train_total_energy": total_e, "exploratory_config": pc["exploratory"],
                               "orthogonal_decomposition_sse": "see report (exact three-term identity)"})
            LOG.info(f"  HoSVD train bound: SSE={h_tr['sse']:.6e} <= {bound:.6e}  (slack {(bound - h_tr['sse']) / bound:.3%})  passed={passed}")
            if run_pca:
                p_tr = split_results[("pca", cid, "train")]
                opt = p_tr["sse"] <= h_tr["sse"] + tol
                enforced = solver == "exact-gram"
                if enforced:
                    tail_pred = float(wtail[k])
                    LOG.info(f"  PCA optimality (TRAIN): PCA SSE={p_tr['sse']:.6e} <= HoSVD SSE={h_tr['sse']:.6e} -> {opt}; "
                             f"PCA SSE vs sum of discarded eigenvalues {tail_pred:.6e} (abs diff {abs(p_tr['sse'] - tail_pred):.2e}, rel to ||Z||^2 {abs(p_tr['sse'] - tail_pred) / total_e:.2e})")
                else:
                    tail_pred = float("nan")
                    LOG.info(f"  PCA(approx fallback) train SSE={p_tr['sse']:.6e}; inequality vs HoSVD reported but NOT enforced ({opt})")
                bound_rows[-1].update({"pca_train_sse": p_tr["sse"], "pca_sum_discarded_eigenvalues": tail_pred,
                                       "pca_sse_vs_discarded_abs_diff": abs(p_tr["sse"] - tail_pred) if enforced else "NA",
                                       "pca_le_hosvd_train_sse": opt, "pca_optimality_enforced": enforced,
                                       "pca_optimality_ratio_pca_over_hosvd": p_tr["sse"] / h_tr["sse"]})
                if enforced and not opt:
                    LOG.info("  *** SERIOUS FLAG: exact PCA train SSE exceeds HoSVD train SSE beyond tolerance ***")
            status_rows.append({"config_id": cid, "k": k, "status": "COMPLETED", "reason": ""})
            completed.append(cid)

        # ---------------- 6. freeze / reload check ----------------
        fitted_arrays["config_ids"] = np.array(completed)
        np.savez(out / "fitted_center_and_factors.npz", **fitted_arrays)
        reload_checks = {}
        with np.load(out / "fitted_center_and_factors.npz") as z:
            for cid in completed:
                m2 = HoSVDModel(z["M"], z[f"{cid}_Q_time"], z[f"{cid}_Q_station"], z[f"{cid}_Q_feature"])
                Xv = data["validation"]["X"][:512]
                with np.load(out / f"features_validation_hosvd_{cid}.npz") as fz:
                    ref = fz["F"][:512]
                new = m2.encode(Xv).astype(np.float32)
                reload_checks[f"hosvd_{cid}"] = float(np.max(np.abs(new - ref)))
                if cid in pca_models:
                    k = pca_models[cid].k
                    with np.load(out / f"fitted_pca_basis_{k}.npz") as pz:
                        m3 = PCAModel(pz["M"], pz["V"])
                    with np.load(out / f"features_validation_pca_{cid}.npz") as fz:
                        ref = fz["F"][:512]
                    reload_checks[f"pca_{cid}"] = float(np.max(np.abs(m3.encode(Xv).astype(np.float32) - ref)))
        LOG.info(f"[freeze/reload] max |F_reload - F_exported| on 512 validation rows: {reload_checks}")

        # ---------------- 7. tables ----------------
        cmp_fields = ["split", "method", "config_id", "k", "r_time", "r_station", "r_feature", "relative_reconstruction_error", "mse",
                      "retained_energy", "fit_seconds", "transform_seconds", "inverse_seconds", "model_bytes", "feature_bytes",
                      "total_estimated_bytes", "peak_ram_mb"]
        write_csv(out / "pca_hosvd_comparison.csv", comp_rows, cmp_fields + [k_ for k_ in comp_rows[0] if k_ not in cmp_fields])
        write_csv(out / "compression_storage.csv", stor_rows,
                  ["split", "method", "config_id", "k", "raw_logical_bytes", "model_parameter_count", "model_bytes", "feature_bytes",
                   "total_estimated_bytes", "dimension_ratio", "effective_storage_ratio", "model_dtype", "feature_dtype",
                   "mean_bytes", "basis_bytes", "code_bytes_per_scalar", "model_bytes_per_scalar", "actual_feature_file_bytes", "exploratory_config"])
        write_csv(out / "hosvd_truncation_bound.csv", bound_rows,
                  ["config_id", "r_time", "r_station", "r_feature", "k", "empirical_train_sse", "tail_time", "tail_station", "tail_feature",
                   "upper_bound_sse", "numerical_tolerance", "bound_passed", "margin", "relative_slack", "sse_over_bound",
                   "pca_train_sse", "pca_sum_discarded_eigenvalues", "pca_sse_vs_discarded_abs_diff", "pca_le_hosvd_train_sse",
                   "pca_optimality_enforced", "pca_optimality_ratio_pca_over_hosvd", "train_total_energy", "exploratory_config"])
        write_csv(out / "phase4_feature_manifest.csv", manifest_rows,
                  ["split", "method", "config_id", "k", "r_time", "r_station", "r_feature", "relative_path", "sha256", "n_samples",
                   "n_features", "metadata_alignment_passed", "actual_file_bytes", "sha256_note"])
        write_csv(out / "phase4_run_status.csv", status_rows)
        write_csv(out / "pca_k_energy_summary.csv", kenergy_rows)
        fallback_rows = []

        # ---------------- 8. optional fallback cross-check (exact run only) ----------------
        if solver == "exact-gram" and args.fallback_crosscheck:
            kk = 48
            LOG.info(f"[crosscheck] approximate randomized fallback vs exact PCA at k={kk} (additional diagnostic; NOT used for any result)")
            Vf, thf, dgf = fit_pca_randomized(Xtr, M, kk, args.batch_size, args.seed)
            Vf = canonicalize_signs(Vf)
            Vex = V_full[:, :kk]
            relerr = np.abs(thf - w_pca[:kk]) / w_pca[:kk]
            pm = PCAModel(M, Vf)
            r_fb, _ = evaluate_split(pm, Xtr, args.batch_size)
            ex_sse = float(tail_energies(w_pca)[kk])
            fallback_rows.append({"object": "pca_fallback_vs_exact", "config_id": "A", "dimension": kk, "comparison": "randomized_vs_exact_gram",
                                  "projector_frobenius_distance": projector_distance(Vf, Vex),
                                  "max_principal_angle_deg": max_principal_angle_deg(Vf, Vex),
                                  "max_rel_eigenvalue_error": float(relerr.max()), "max_rel_ritz_residual": dgf["max_rel_ritz_residual_top_k"],
                                  "train_sse_fallback": r_fb["sse"], "train_sse_exact_tail_sum": ex_sse,
                                  "train_sse_rel_excess": (r_fb["sse"] - ex_sse) / ex_sse, "seed": args.seed, "n_iter": dgf["n_iter"],
                                  "oversample": dgf["oversample"], "seconds": dgf["seconds"]})
            LOG.info(f"  {fallback_rows[-1]}")
            meta["fallback_crosscheck"] = fallback_rows[-1]
        write_csv(out / "stability_diagnostics.csv", stab_rows + fallback_rows)

        # ---------------- 9. figures ----------------
        make_figures(out)

        # ---------------- 10. metadata ----------------
        meta.update({"configs": status_rows, "completed_configs": completed, "pca_info": {k_: v for k_, v in pca_info.items()},
                     "train_total_energy": total_e, "raw_dim": d, "freeze_reload_max_abs_diff": reload_checks,
                     "direct_path_checks": direct_checks, "peak_rss_mb_after_gram": rss_after_gram, "peak_rss_mb_after_fit": fit_peak,
                     "peak_rss_mb_final": peak_rss_mb(), "wall_seconds_total": now() - t_start,
                     "train_only_artifacts_not_counted_in_deployment_storage": {
                         "flat_gram_float64_bytes": d * d * 8, "full_pca_eigenbasis_float64_bytes": d * d * 8,
                         "mode_grams_and_full_mode_bases_bytes": (24 * 24 + 12 * 12 + 11 * 11) * 8 * 2},
                     "metric_definition": "Primary metrics use explicit reconstruction from full-precision (float64) internal codes; "
                                          "float32-exported-code metrics are given in *_float32_codes columns.",
                     "status": "COMPLETED"})
        json.dump(meta, open(out / "phase4_metadata.json", "w"), indent=2, default=str)
        LOG.info(f"[done] completed configs {completed}; wall {now() - t_start:.1f}s; peak RSS {peak_rss_mb():.0f} MB")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------------
def make_figures(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(out)
    comp = read_csv(out / "pca_hosvd_comparison.csv")
    stor = read_csv(out / "compression_storage.csv")
    bound = read_csv(out / "hosvd_truncation_bound.csv")
    modes = read_csv(out / "hosvd_mode_spectra.csv")
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)

    # fig1: RRE vs dimension
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for a, s in zip(ax, SPLITS):
        for method, mk, col in [("pca", "o-", "tab:blue"), ("hosvd", "s--", "tab:red")]:
            rows = [r for r in comp if r["split"] == s and r["method"] == method and r["exploratory_config"] == "False"]
            rows.sort(key=lambda r: int(r["k"]))
            a.plot([int(r["k"]) for r in rows], [float(r["relative_reconstruction_error"]) for r in rows], mk, color=col,
                   label="PCA/SVD" if method == "pca" else "HoSVD (shared factors)")
        for r in [r for r in comp if r["split"] == s and r["method"] == "hosvd" and r["exploratory_config"] == "True"]:
            a.plot(int(r["k"]), float(r["relative_reconstruction_error"]), "^", color="tab:green", label="HoSVD alt allocation (exploratory)")
        a.set_xscale("log"); a.set_title(f"{s} (N={comp[[r['split'] for r in comp].index(s)]['n_samples']})")
        a.set_xlabel("coefficients per sample k"); a.grid(alpha=.3)
        h, l = a.get_legend_handles_labels(); u = dict(zip(l, h)); a.legend(u.values(), u.keys(), fontsize=8)
    ax[0].set_ylabel("relative reconstruction error (RRE)")
    fig.suptitle("Fig 1. RRE = ||Z-Zhat||_F/||Z||_F (Phase-2 standardized units) vs k; bases fitted on TRAIN only, common train mean M")
    fig.tight_layout(); fig.savefig(fig_dir / "fig1_error_vs_dimension.png", dpi=140); plt.close(fig)

    # fig2: train SSE vs bound (same squared-error scale)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ids = [b["config_id"] for b in bound]
    xs = np.arange(len(ids))
    emp = [float(b["empirical_train_sse"]) for b in bound]
    ub = [float(b["upper_bound_sse"]) for b in bound]
    ax.bar(xs - 0.2, emp, 0.4, label="empirical HoSVD train SSE (squared Frobenius)", color="tab:red")
    ax.bar(xs + 0.2, ub, 0.4, label="bound: tail_time+tail_station+tail_feature", color="tab:gray")
    pc = [(i, float(b["pca_train_sse"])) for i, b in enumerate(bound) if b.get("pca_train_sse") not in (None, "")]
    if pc:
        ax.plot([i for i, _ in pc], [v for _, v in pc], "ko", label="exact PCA train SSE (same k)")
    ax.set_xticks(xs); ax.set_xticklabels([f"{b['config_id']}\n({b['r_time']},{b['r_station']},{b['r_feature']}) k={b['k']}" for b in bound], fontsize=8)
    ax.set_ylim(bottom=0); ax.set_ylabel("squared Frobenius error on TRAIN (standardized units^2)"); ax.grid(alpha=.3, axis="y")
    ax.set_title("Fig 2. TRAIN HoSVD squared error vs truncation bound (same squared scale, linear axis)\nthe bound is derived from TRAIN spectra and is not a validation/test guarantee", fontsize=10); ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(fig_dir / "fig2_train_hosvd_error_bound.png", dpi=140); plt.close(fig)

    # fig3: storage vs dimension (train split)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for method, col, mk in [("pca", "tab:blue", "o-"), ("hosvd", "tab:red", "s--")]:
        rows = [r for r in stor if r["split"] == "train" and r["method"] == method and r["exploratory_config"] == "False"]
        rows.sort(key=lambda r: int(r["k"]))
        ks = [int(r["k"]) for r in rows]
        ax[0].plot(ks, [float(r["model_bytes"]) for r in rows], mk, color=col, label=f"{method} model bytes (M + basis, float64)")
        ax[1].plot(ks, [float(r["total_estimated_bytes"]) for r in rows], mk, color=col, label=f"{method} total = codes + model")
    raw = float([r for r in stor if r["split"] == "train"][0]["raw_logical_bytes"])
    ax[1].axhline(raw, color="k", ls=":", label="raw float32 train tensor")
    for a, t in zip(ax, ["model storage (basis + mean)", "total estimated storage, TRAIN split"]):
        a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("k"); a.set_ylabel("bytes"); a.set_title(t); a.grid(alpha=.3); a.legend(fontsize=8)
    fig.suptitle("Fig 3. Storage vs dimension: equal k is NOT equal storage")
    fig.tight_layout(); fig.savefig(fig_dir / "fig3_storage_vs_dimension.png", dpi=140); plt.close(fig)

    # fig4: mode spectra
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for a, nm in zip(ax, ["time", "station", "feature"]):
        rows = [r for r in modes if r["mode"] == nm]
        a.semilogy([int(r["index"]) for r in rows], [max(float(r["singular_value"]), 1e-12) for r in rows], "o-")
        for cid, (rk, _) in CONFIGS.items():
            if cid in ("A", "B", "C"):
                a.axvline(rk[["time", "station", "feature"].index(nm)] + 0.5, ls=":", alpha=.6)
        a.set_title(f"mode: {nm}"); a.set_xlabel("index j"); a.set_ylabel("sqrt(eigenvalue of mode Gram)"); a.grid(alpha=.3)
    fig.suptitle("Fig 4. Train mode singular spectra (dotted: truncation ranks of A/B/C)")
    fig.tight_layout(); fig.savefig(fig_dir / "fig4_mode_singular_spectra.png", dpi=140); plt.close(fig)

    # fig5: PCA spectrum
    sp = read_csv(out / "pca_components_spectrum.csv")
    if sp and sp[0].get("status", "").startswith("complete"):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        ax[0].semilogy([int(r["component"]) for r in sp], [max(float(r["singular_value"]), 1e-12) for r in sp])
        ax[1].semilogy([int(r["component"]) for r in sp], [max(float(r["tail_energy_fraction_after_j"]), 1e-16) for r in sp])
        for k in (48, 192, 576):
            for a in ax:
                a.axvline(k, ls=":", color="gray")
        ax[0].set_title("flat PCA singular values (from float64 Gram)"); ax[1].set_title("PCA tail energy fraction after j (dotted: k=48,192,576)")
        for a in ax:
            a.set_xlabel("component j"); a.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(fig_dir / "fig5_pca_spectrum.png", dpi=140); plt.close(fig)

    # fig6: stability
    stab = out / "stability_diagnostics.csv"
    if stab.exists():
        rows = [r for r in read_csv(stab) if r.get("projector_frobenius_distance") not in ("", None) and r.get("comparison") in
                ("scipy_vs_numpy_eigh_same_gram", "split_half_train_first_vs_second")]
        if rows:
            fig, ax = plt.subplots(figsize=(10, 4.5))
            labels = sorted({(r["object"], r["config_id"]) for r in rows})
            x = np.arange(len(labels))
            for off, comp_ in [(-0.2, "scipy_vs_numpy_eigh_same_gram"), (0.2, "split_half_train_first_vs_second")]:
                vals = []
                for lab in labels:
                    v = [float(r["projector_frobenius_distance"]) for r in rows if (r["object"], r["config_id"]) == lab and r["comparison"] == comp_]
                    vals.append(v[0] if v else np.nan)
                ax.bar(x + off, np.maximum(vals, 1e-18), 0.4, label=comp_)
            ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([f"{a}\n{b}" for a, b in labels], fontsize=7, rotation=0)
            ax.set_ylabel("||QQ^T - Q_altQ_alt^T||_F"); ax.grid(alpha=.3, axis="y"); ax.legend(fontsize=8)
            ax.set_title("Fig 6. Subspace (projector) stability diagnostics")
            fig.tight_layout(); fig.savefig(fig_dir / "fig6_subspace_stability.png", dpi=140); plt.close(fig)


# ----------------------------------------------------------------------------------------
# report + package  (re-runnable from the CSV/JSON outputs: --finalize)
# ----------------------------------------------------------------------------------------
def fmt(x, p=6):
    try:
        return f"{float(x):.{p}f}"
    except Exception:
        return str(x)


def sci(x, p=4):
    try:
        return f"{float(x):.{p}e}"
    except Exception:
        return str(x)


def text_table(header, body):
    body = [[str(c) for c in r] for r in body]
    w = [max(len(str(h)), *(len(r[i]) for r in body)) if body else len(str(h)) for i, h in enumerate(header)]
    line = "  ".join(str(h).ljust(w[i]) for i, h in enumerate(header))
    out = [line, "  ".join("-" * x for x in w)]
    out += ["  ".join(r[i].ljust(w[i]) for i in range(len(header))) for r in body]
    return "\n".join(out)


def write_report(out, toy_p, toy_h):
    out = Path(out)
    meta = json.load(open(out / "phase4_metadata.json"))
    comp = read_csv(out / "pca_hosvd_comparison.csv")
    stor = read_csv(out / "compression_storage.csv")
    bound = read_csv(out / "hosvd_truncation_bound.csv")
    modes = read_csv(out / "hosvd_mode_spectra.csv")
    kener = read_csv(out / "pca_k_energy_summary.csv")
    stab = read_csv(out / "stability_diagnostics.csv")
    man = read_csv(out / "phase4_feature_manifest.csv")
    status = read_csv(out / "phase4_run_status.csv")
    pt = out / "pytest_results.txt"
    pytest_txt = pt.read_text() if pt.exists() else "pytest_results.txt NOT PRESENT: tests were not run before finalize."
    cfgs = [c["config_id"] for c in status if c["status"] == "COMPLETED"]
    main_cfgs = [c for c in cfgs if not CONFIGS[c][1]]
    expl_cfgs = [c for c in cfgs if CONFIGS[c][1]]
    notrun = [c for c in status if c["status"] != "COMPLETED"]

    def get(split, method, cid):
        r = [x for x in comp if x["split"] == split and x["method"] == method and x["config_id"] == cid]
        return r[0] if r else None

    L = []
    A = L.append
    A("PHASE 4 -- RESULTS AND MATHEMATICS")
    A("Matched-dimensional matrix PCA/SVD versus shared-basis truncated HoSVD (unsupervised reconstruction analysis)")
    A("=" * 110)
    A("")
    A("LEGEND.  [MEASURED] = produced by executing phase4_pca_hosvd.py on the supplied Phase 2 arrays in this run.")
    A("          [DERIVED]  = follows analytically from the definitions (proof sketch given) or from arithmetic on measured numbers.")
    A("          [TOY]      = hand-worked illustrative example, NOT a Beijing finding.")
    A("Scope: no SVM / RFF / Markov / classification metric was computed; no configuration was selected using validation or test data.")
    A("All reconstruction errors are in Phase-2 STANDARDIZED feature units, not ug/m^3.")
    A("")
    A("0. EXECUTION STATUS  [MEASURED]")
    A("-" * 110)
    A(f"Run status: {meta['status']}.  Completed configurations: {', '.join(cfgs)}.")
    A(f"Mandatory A and B: {'completed' if all(c in cfgs for c in ('A', 'B')) else 'NOT COMPLETED'}.  "
      f"C: {'completed' if 'C' in cfgs else 'NOT RUN'}.  Exploratory (additional, lower-priority) allocations run: {', '.join(expl_cfgs) if expl_cfgs else 'none'}.")
    for c in notrun:
        A(f"  NOT RUN: config {c['config_id']} k={c['k']}: {c['reason']}")
    A(f"PCA solver used: {meta['pca_solver']['used']} ({meta['pca_solver']['reason']}). Requested: {meta['pca_solver']['requested']}.")
    A(f"Environment: {meta['environment'].get('platform')}; CPU {meta['environment'].get('cpu_model')} x{meta['environment'].get('cpu_count_logical')} logical; "
      f"RAM {meta['environment'].get('ram_total_gb')} GB; Python {meta['environment']['python'].split()[0]}, NumPy {meta['environment']['numpy']}, SciPy {meta['environment']['scipy']}.")
    A(f"Seed {meta['seed']} (the exact-Gram path is deterministic; the seed only drives the randomized fallback cross-check and the audit-subset selection).")
    A(f"Script sha256 {meta['code_sha256']}")
    A("Hashes are stable within a fixed software environment; cross-platform byte-for-byte identity is NOT claimed.")
    A("")
    A("1. INPUT CONTRACT  [MEASURED]")
    A("-" * 110)
    for sp in SPLITS:
        r = meta["input_validation"][sp]
        A(f"{sp:10s} N={r['N']}  X{tuple(r['X_shape'])} float32 finite; class counts {r['class_counts']}; current_state==-1: {r['n_current_state_unknown']} "
          f"(current_pm25 NaN iff state==-1: {r['nan_iff_unknown_state']}); targets {r['t_first'][:19]} .. {r['t_last'][:19]}; raw logical bytes {r['logical_float32_bytes']}")
    A(f"Cross-split: strictly chronological, no timestamp overlap, no duplicates inside splits: {meta['cross_split']}.")
    A("Unknown-current-state rows (-1) were KEPT (treated as metadata). y was never used by any decomposition.")
    A("Input SHA-256:")
    for k, v in meta["inputs"]["sha256"].items():
        A(f"   {k}  {v}")
    if "zip_sha256" in meta["inputs"]:
        A(f"   [input ZIP] {meta['inputs']['source']}  {meta['inputs']['zip_sha256']}")
    A("   (training_preprocessing_parameters.npz, phase2_summary.json, PHASE2_README.txt are informational only: hashed, never refit.)")
    A("")
    A("2. DATA DIAGNOSTIC: DUPLICATED METEOROLOGICAL CHANNELS (identical to float32 round-off)  [MEASURED]")
    A("-" * 110)
    dup = meta.get("identical_channel_groups_train_within_1e-6", {})
    if dup:
        A("In the TRAIN tensor several (station, measurement) series are copies of each other (max |difference| <= 1e-6, i.e. float32 round-off level):")
        for feat, groups in dup.items():
            A(f"   {feat:6s}: " + " | ".join("{" + ", ".join(g) + "}" for g in groups))
        A(f"Redundant station-feature channels: {meta['n_redundant_station_feature_channels']} ({meta['n_redundant_channels_bitwise']} of them bitwise identical; the others, PRES pairs, differ from their twin by one float32 ulp, ~3e-8, at a single hour).")
        A(f"Expected (near-)null directions of the flat Gram = redundant channels x 24 hours = {meta['implied_exact_null_directions_of_flat_gram']}.")
        A(f"Observed flat-Gram eigenvalues <= tolerance: {meta['pca_info'].get('n_below_tol')}; numerical rank (eigenvalue > tolerance): {meta['pca_info'].get('numerical_rank_above_tol')} of {meta['raw_dim']}.")
        A("Consequences (stated, not tuned away): (i) the flat Gram is rank deficient, so tiny/negative eigenvalues are expected rather than a solver error; (ii) duplicated channels receive multiplicity weight in every")
        A("Frobenius-norm quantity (PCA, HoSVD and the error metrics alike); (iii) nothing was removed or re-weighted, because Phase 2 is frozen.  This is a property of the supplied Phase 2 arrays, not of PCA/HoSVD.")
    else:
        A("No duplicate-channel diagnostic present in metadata (older run).")
    A("")
    A("3. SETUP AND DEFINITIONS  [DERIVED]")
    A("-" * 110)
    A("Raw dimension d = 24 x 12 x 11 = 3168 per sample.  Training mean tensor M[a,b,c] = (1/N_train) sum_i X[i,a,b,c]  (float64; stored).  Z_i = X_i - M for every split.")
    A("Matrix PCA/SVD: A = flatten_C(Z_train) in R^{24108 x 3168} (C order: index = a*132 + b*11 + c).  G = A^T A (float64, streamed in batches).  Eigen-decompose G = V diag(lambda) V^T,")
    A("  lambda_1 >= ... ; singular values sigma_j ~ sqrt(max(lambda_j,0)).  Feature f_PCA = vec(Z_i) V_k;  reconstruction X_hat = M + reshape(f V_k^T).")
    A("Shared HoSVD (sample mode uncompressed): mode Grams G_time (24x24), G_station (12x12), G_feature (11x11) accumulated over all training samples; leading eigenvectors give Q_time, Q_station, Q_feature.")
    A("  Core C_i = Z_i x_time Q_time^T x_station Q_station^T x_feature Q_feature^T  (shape r_t x r_s x r_f);  F_i = vec_C(C_i) with index p*(r_s*r_f) + q*r_f + r.")
    A("  Inverse Zhat_i = C_i x_time Q_time x_station Q_station x_feature Q_feature;  X_hat_i = M + Zhat_i.  Einsum index order: 'iabc,ap->ipbc', 'ipbc,bq->ipqc', 'ipqc,cr->ipqr'.")
    A("  Equivalent flat form: vec(C_i) = vec(Z_i) K with K = Q_time (x) Q_station (x) Q_feature (Kronecker, d x k, orthonormal columns); this was used as the independent direct path in tests/audit subsets.")
    A("Sign convention: each factor column is flipped so its largest-|.| entry is non-negative, BEFORE any feature export.")
    A("Metrics: SSE = ||Z-Zhat||_F^2, RRE = sqrt(SSE/||Z||_F^2), MSE = SSE/(N*3168), retained_energy = ||Zhat||_F^2/||Z||_F^2, all accumulated in float64 per split.")
    A("  PRIMARY metrics use explicit reconstruction from FULL-PRECISION (float64) internal codes.  The *_float32_codes columns repeat the metric decoding the float32-EXPORTED codes.")
    A("  The float32 and float64 RREs differ by at most " + sci(max(abs(float(r['rre_float32_minus_float64'])) for r in comp), 2) + " (absolute)  [MEASURED], i.e. immaterial for every reported digit.")
    A("")
    A("4. THEOREMS USED  [DERIVED]")
    A("-" * 110)
    A("T1 (projection identity).  P = K K^T with K orthonormal (PCA: K=V_k; HoSVD: K=Kronecker factor).  Then Z - PZ is orthogonal to PZ, so SSE = ||Z||^2 - ||PZ||^2 = ||Z||^2 - ||codes||^2, hence RRE^2 + retained_energy = 1.")
    A("   Float32 storage of F breaks this only at ~1e-7 relative level in the stored codes; the code verifies the identity in float64 against explicit reconstruction.")
    A("T2 (mode eigenvalues).  The mode-t Gram G_t = Z_(t) Z_(t)^T where Z_(t) is the mode-t unfolding of the whole training collection (sample mode appended to the columns).  Its eigenvalues are the squared")
    A("   singular values of Z_(t); trace(G_t) = ||Z_train||_F^2 for all three modes and for the flat Gram (all verified numerically).")
    A("T3 (HoSVD training truncation bound).  Let P_t, P_s, P_f be the orthogonal projectors Q Q^T acting on the time, station and feature modes.  They act on different tensor modes, so they commute, are")
    A("   self-adjoint and idempotent in the Frobenius inner product, and Zhat = P_t P_s P_f Z.  Telescoping,")
    A("       Z - Zhat = (I-P_t)Z  +  P_t (I-P_s) Z  +  P_t P_s (I-P_f) Z.")
    A("   The three terms are mutually orthogonal (e.g. <(I-P_t)Z, P_t W> = <Z,(I-P_t)P_t W> = 0, and similarly using (I-P_s)P_s = 0 and commutation), hence")
    A("       ||Z-Zhat||^2 = ||(I-P_t)Z||^2 + ||P_t (I-P_s)Z||^2 + ||P_t P_s (I-P_f)Z||^2  <=  ||(I-P_t)Z||^2 + ||(I-P_s)Z||^2 + ||(I-P_f)Z||^2,")
    A("   because projectors are contractions.  Finally ||(I-P_t)Z||^2 = ||Z_(t)||^2 - ||Q_t^T Z_(t)||^2 = sum_{j>r_t} lambda_j(G_t) = tail_time (Q_t spans the top-r_t eigenspace of G_t); same for the other modes.")
    A("       ||Z_train - Zhat_train||_F^2  <=  tail_time(r_t) + tail_station(r_s) + tail_feature(r_f).   (classical truncated-HOSVD argument; sample mode uncompressed)")
    A("   The first term equals tail_time EXACTLY; the other two are bounded by their tails.  The bound depends on TRAIN spectra and is NOT claimed for validation/test.")
    A("T4 (PCA optimality on TRAIN).  By Eckart-Young-Mirsky / Ky Fan, among all rank-k orthogonal projectors P on R^d, ||A - A P||_F^2 is minimized by the span of the top-k right singular vectors,")
    A("   with minimum sum_{j>k} lambda_j.  HoSVD reconstructs flat(Z) K K^T, a projection onto a particular k-dimensional (Kronecker-structured) subspace; therefore")
    A("   SSE_PCA,train(k) <= SSE_HoSVD,train(k) for exact PCA at equal k.  This is a statement about TRAIN reconstruction of the flattened training matrix only; it implies nothing about validation/test reconstruction or classification.")
    A("T5 (numerics of the Gram route).  Eigenvalues of the float64 Gram are exact for the COMPUTED Gram, not an infinite-precision SVD of A; forming A^T A squares the condition number, so eigenvalues")
    A("   below roughly n*eps*lambda_max are not resolved.  Declared policy: tol = n*eps64*lambda_max; eigenvalues < -tol raise an error; eigenvalues in [-tol,0) are clipped to 0 and counted/logged.")
    A("")
    A("5. TOY EXAMPLES  [TOY -- illustrative, NOT Beijing findings]")
    A("-" * 110)
    A("2x2 matrix, rank-1 PCA: A = [[2,1],[1,2]].  Gram A^T A = [[5,4],[4,5]], eigenvalues 9 and 1 (singular values 3 and 1), v1 = (1,1)/sqrt(2).")
    A(f"   computed: eigenvalues {np.round(toy_p['eigenvalues'], 12).tolist()}, v1={np.round(toy_p['v1'], 12).tolist()}, rank-1 reconstruction {toy_p['rank1_recon'].tolist()}, SSE={toy_p['sse']:.12f}, retained fraction={toy_p['retained_fraction']:.12f}")
    A("   hand: reconstruction rows both (1.5,1.5); residual rows (0.5,-0.5),(-0.5,0.5); SSE = 1 = discarded eigenvalue; retained 9/10.")
    A("2x2x2 tensor T with T[0,0,0]=2, T[1,1,1]=1 (all other entries 0), rank-(1,1,1) HOSVD (all three modes compressed in this toy):")
    A("   every mode unfolding is a 2x4 matrix with Gram diag(4,1) (singular values 2,1); leading eigenvectors e_0; core = 2; approximation 2*e0(x)e0(x)e0.")
    A(f"   computed: SSE={toy_h['sse']:.12f}, tails={[round(t, 12) for t in toy_h['tails']]}, bound={toy_h['bound']:.12f}.  Hand: error = |T[1,1,1]|^2 = 1 <= 1+1+1 = 3 (the bound holds and is not tight).")
    A("")
    A("6. RECONSTRUCTION RESULTS  [MEASURED]   (bases fitted on TRAIN only; validation/test are held-out diagnostics)")
    A("-" * 110)
    hdr = ["config", "ranks", "k", "d/k", "method", "RRE_train", "RRE_val", "RRE_test", "retE_train", "retE_val", "retE_test"]
    body = []
    for cid in cfgs:
        for m in ("pca", "hosvd"):
            rr = [get(s, m, cid) for s in SPLITS]
            if rr[0] is None:
                continue
            ranks = "-" if m == "pca" else f"({rr[0]['r_time']},{rr[0]['r_station']},{rr[0]['r_feature']})"
            body.append([cid + (" (expl.)" if CONFIGS[cid][1] else ""), ranks, rr[0]["k"], fmt(3168 / int(rr[0]["k"]), 2), m.upper() if m == "pca" else "HoSVD"] +
                        [fmt(r["relative_reconstruction_error"], 6) for r in rr] + [fmt(r["retained_energy"], 6) for r in rr])
    A(text_table(hdr, body))
    A("")
    hdr = ["config", "method", "MSE_train", "MSE_val", "MSE_test", "SSE_train", "||Z||^2/(N*d) train", "val", "test"]
    body = []
    for cid in cfgs:
        for m in ("pca", "hosvd"):
            rr = [get(s, m, cid) for s in SPLITS]
            if rr[0] is None:
                continue
            body.append([cid, m, *[fmt(r["mse"], 6) for r in rr], sci(rr[0]["sse"], 5),
                         *[fmt(float(r["znorm2"]) / (int(r["n_samples"]) * 3168), 5) for r in rr]])
    A(text_table(hdr, body))
    A("   (MSE is per feature entry in standardized units^2.  The last three columns show the per-entry centered energy of each split: it differs between splits, which is why a relative error")
    A("    (RRE) of a held-out split can be lower than the training RRE without any 'better generalization'.)")
    A("")
    A("PCA versus HoSVD at equal k, [MEASURED] differences in RRE (HoSVD - PCA; positive = PCA reconstructs better):")
    hdr = ["config", "k", "train", "validation", "test", "RRE_ratio HoSVD/PCA (train / val / test)"]
    body = []
    npca_better = 0
    ntot = 0
    for cid in main_cfgs:
        dd, rat = [], []
        for s in SPLITS:
            a_, b_ = get(s, "pca", cid), get(s, "hosvd", cid)
            x, y = float(a_["relative_reconstruction_error"]), float(b_["relative_reconstruction_error"])
            dd.append(fmt(y - x, 6)); rat.append(fmt(y / x, 3)); ntot += 1; npca_better += int(x < y)
        body.append([cid, get("train", "pca", cid)["k"], *dd, " / ".join(rat)])
    A(text_table(hdr, body))
    A(f"   PCA had the lower RRE in {npca_better} of {ntot} (config x split) comparisons among the declared configurations A/B/C.  This is a descriptive observation about reconstruction only; ")
    A("   it is NOT a statement about next-state classification (Phase 5), and HoSVD is not claimed to be 'better' or 'worse' overall (storage and cost differ, see Sections 8-9).")
    A("")
    A("Retained energy as fractions of the TRAIN total energy [MEASURED]: PCA k-energy summary")
    for r in kener:
        if r["method"] == "pca":
            A(f"   PCA k={r['k']:>4s} ({r['config_id']}): retained {fmt(r['retained_energy_fraction'], 6)}, tail {fmt(r['tail_energy_fraction'], 6)} (= 1 - retained; train SSE/||Z||^2)")
    A("")
    A("Mode energy retained by the shared-factor ranks (TRAIN mode spectra; fraction of ||Z||_F^2 captured by the leading r eigenvectors of each mode Gram) [MEASURED]:")
    hdr = ["config", "ranks", "time", "station", "feature"]
    body = []
    for cid in cfgs:
        r = CONFIGS[cid][0]
        vals = []
        for nm, rk in zip(["time", "station", "feature"], r):
            mm = [x for x in modes if x["mode"] == nm and int(x["index"]) == rk][0]
            vals.append(fmt(mm["retained_energy_fraction"], 4))
        body.append([cid + (" (expl.)" if CONFIGS[cid][1] else ""), str(r), *vals])
    A(text_table(hdr, body))
    A("   Reading: the three single-mode retained fractions are DIFFERENT quantities from the joint reconstruction retained_energy above (the joint retention of a Kronecker projector is not the product of mode fractions).")
    A("   Roles: the time mode summarises temporal smoothness/trend over the 24-h window (its spectrum decays smoothly, Fig 4), the station mode summarises spatial co-variation across the 12 sites (a dominant first component,")
    A("   i.e. a city-wide common level), and the feature mode summarises the correlations among the 11 pollutants/meteorological variables.  Extra rank in a mode only buys the energy of that mode's next eigenvalues.")
    A("")
    if expl_cfgs:
        A("Exploratory k=192 alternate rank allocations (ADDITIONAL experiments; reported, never used to pick a configuration) [MEASURED]:")
        hdr = ["alloc", "ranks", "RRE_train", "RRE_val", "RRE_test", "train bound slack"]
        body = []
        for cid in ["B"] + expl_cfgs:
            if cid not in cfgs:
                continue
            rr = [get(s, "hosvd", cid) for s in SPLITS]
            b = [x for x in bound if x["config_id"] == cid][0]
            body.append([cid, str(CONFIGS[cid][0]), *[fmt(r["relative_reconstruction_error"], 6) for r in rr], f"{float(b['relative_slack']):.3%}"])
        A(text_table(hdr, body))
        A("   Allocation changes the retained information at fixed k, so rank allocation matters; since these were not pre-registered as selection candidates and no selection rule was applied, none is declared 'chosen'.")
        A("   Any later adoption of an allocation should be decided on TRAIN/validation criteria in Phase 5 design, not on these test numbers.")
        A("")
    A("7. TRAIN CHECKS  [MEASURED]: HoSVD truncation bound (T3) and exact-PCA optimality (T4)")
    A("-" * 110)
    hdr = ["config", "ranks", "k", "HoSVD SSE", "bound=sum tails", "slack", "passed", "PCA SSE", "PCA sum discarded eig", "|diff|", "PCA<=HoSVD", "PCA/HoSVD SSE"]
    body = []
    for b in bound:
        body.append([b["config_id"], f"({b['r_time']},{b['r_station']},{b['r_feature']})", b["k"], sci(b["empirical_train_sse"], 5), sci(b["upper_bound_sse"], 5),
                     f"{float(b['relative_slack']):.3%}", b["bound_passed"], sci(b["pca_train_sse"], 5) if b.get("pca_train_sse") else "n/a",
                     sci(b["pca_sum_discarded_eigenvalues"], 5) if b.get("pca_sum_discarded_eigenvalues") not in ("", None) else "n/a",
                     sci(b["pca_sse_vs_discarded_abs_diff"], 2) if b.get("pca_sse_vs_discarded_abs_diff") not in ("", None, "NA") else "n/a",
                     b.get("pca_le_hosvd_train_sse") or "n/a", fmt(b["pca_optimality_ratio_pca_over_hosvd"], 4) if b.get("pca_optimality_ratio_pca_over_hosvd") else "n/a"])
    A(text_table(hdr, body))
    A(f"   Declared tolerance: {sci(bound[0]['numerical_tolerance'], 3)} = {ENERGY_REL_TOL:g} x ||Z_train||_F^2 = {ENERGY_REL_TOL:g} x {sci(bound[0]['train_total_energy'], 6)} (squared standardized units).")
    A("   'slack' = (bound - empirical)/bound.  Both quantities are on the SAME squared-error scale (Fig 2).  The bound is TRAIN-specific.")
    A("   PCA SSE vs the sum of discarded eigenvalues is the independent check that the explicit float64 residual matches the spectrum (T4/T5).")
    ident = max(float(r["identity_abs_diff"]) / float(r["znorm2"]) for r in comp)
    A(f"   Projection identity (T1): max over ALL method/config/split rows of |SSE_explicit - (||Z||^2 - ||codes||^2)| / ||Z||^2 = {sci(ident, 2)} (tolerance {IDENTITY_REL_TOL:g}).")
    A(f"   RRE^2 + retained_energy - 1: max |.| = {sci(max(abs(float(r['rre_sq_plus_retained']) - 1) for r in comp), 2)}.")
    dc = meta.get("direct_path_checks", {})
    if dc:
        A(f"   Independent direct path (explicit projector/Kronecker basis, 64-sample subsets per split): max |reconstruction difference| = {sci(max(v['max_abs_recon_diff'] for v in dc.values()), 2)}; "
          f"max SSE-shortcut relative gap = {sci(max(v['shortcut_rel_gap'] for v in dc.values()), 2)}.")
    A(f"   Freeze/reload (fitted npz -> re-encode 512 validation rows vs exported F): max |diff| = {sci(max(meta['freeze_reload_max_abs_diff'].values()), 2)}.")
    A("")
    A("8. STORAGE ACCOUNTING  [DERIVED arithmetic on declared dtypes; on-disk sizes MEASURED]")
    A("-" * 110)
    A("Definitions: feature_bytes = N*k*4 (float32 codes); model_bytes = mean (3168*8) + basis, basis float64 (8 bytes): PCA 3168*k*8; HoSVD (24 r_t + 12 r_s + 11 r_f)*8;")
    A("total_estimated_bytes = feature_bytes + model_bytes; raw_logical_bytes = N*3168*4; effective_storage_ratio = raw/total; dimension_ratio = d/k.  Equal k is NOT equal storage.")
    hdr = ["split", "config", "method", "k", "dim ratio d/k", "model_bytes", "feature_bytes", "total_bytes", "effective ratio"]
    body = []
    for r in stor:
        if r["exploratory_config"] == "True":
            continue
        body.append([r["split"], r["config_id"], r["method"], r["k"], fmt(r["dimension_ratio"], 2), r["model_bytes"], r["feature_bytes"], r["total_estimated_bytes"], fmt(r["effective_storage_ratio"], 3)])
    A(text_table(hdr, body))
    A("   Note: per-split effective ratios include the one-time model bytes, so PCA's larger basis weighs more on the small validation/test splits; dimension_ratio d/k ignores model bytes entirely.")
    A("   Excluded from this deployment-style accounting (training-only artifacts): flat Gram (80,289,792 B), full 3168x3168 PCA eigenbasis (80,289,792 B), mode Grams/full mode bases (small).")
    A("   On-disk npz/ZIP sizes (see compression_storage.csv: actual_feature_file_bytes) reflect deflate on float data and are NOT the mathematical compression ratio.")
    A("")
    A("9. TIMING AND RESOURCES  [MEASURED; single run, single core -- timings vary between runs]")
    A("-" * 110)
    fs = meta["fit_seconds"]
    A(f"One-time fit (wall, shared by all configs of that method): PCA {fs['pca']:.2f} s (mean pass + centering + flat Gram {fs['components']['flat_gram']:.2f} s + eigh {fs['components']['pca_eig']:.2f} s);"
      f" HoSVD {fs['hosvd']:.2f} s (mean pass + centering + mode Grams {fs['components']['mode_gram']:.2f} s + three small eigh {fs['components']['mode_eig_total']:.4f} s).")
    A("   Both Gram accumulations were done in the same streaming pass using separate per-operation timers; the shared mean pass and batch cast/centering are charged to both methods.")
    A("   Note PCA's eigensolve is O(d^3) with d=3168 (dense), HoSVD's three eigensolves are on 24x24, 12x12, 11x11 matrices.")
    hdr = ["config", "method", "split", "transform s (median of 3)", "inverse s (median of 3)", "per-sample multiply-adds (derived, encode)"]
    body = []
    for cid in main_cfgs:
        r_ = CONFIGS[cid][0]
        for m in ("pca", "hosvd"):
            for s in SPLITS:
                r = get(s, m, cid)
                if m == "pca":
                    fl = int(r["k"]) * 3168
                else:
                    fl = r_[0] * 3168 + r_[0] * r_[1] * 132 + r_[0] * r_[1] * r_[2] * 11
                body.append([cid, m, s, fmt(r["transform_seconds"], 4), fmt(r["inverse_seconds"], 4), fl])
    A(text_table(hdr, body))
    A("   transform = cast + center + project + float32 cast for the whole split in batches; inverse = float32 codes -> float64 -> X_hat (adds M) -> float32.  Python/einsum overhead means measured ratios")
    A("   are far smaller than the multiply-add ratios (derived column); neither is a claim about optimized production code.")
    A(f"Peak resident memory (process high-water mark via resource.getrusage; cumulative, so later rows repeat it): {meta['peak_rss_mb_final']:.0f} MB (after Gram stage {meta['peak_rss_mb_after_gram']:.0f} MB; after fit {meta['peak_rss_mb_after_fit']:.0f} MB).")
    A(f"Total wall time {meta['wall_seconds_total']:.1f} s.")
    A("")
    A("10. NUMERICAL DIAGNOSTICS AND REPRODUCIBILITY  [MEASURED]")
    A("-" * 110)
    pi = meta["pca_info"]
    if pi.get("exact"):
        A(f"Flat Gram eigensolve (scipy.linalg.eigh, full 3168 spectrum computed; spectrum CSV is COMPLETE): lambda min raw {sci(pi['min_raw_eigenvalue'], 3)}, tolerance {sci(pi['neg_tol'], 3)}, "
          f"clipped {pi['n_negative_clipped']} tiny negatives (sum |clipped| = {sci(pi['clipped_abs_sum'], 3)}); no significantly negative eigenvalue (none < -tol).")
        A(f"   sum(eigenvalues)/||Z||^2 relative difference {sci(pi['trace_rel_diff_vs_train_energy'], 2)}; Gram symmetry error {sci(pi['asymmetry_rel'], 2)}.")
    for nm, d_ in meta["mode_eig_diag"].items():
        A(f"Mode {nm}: min eigenvalue {sci(d_['min_raw_eigenvalue'], 3)} (> 0, no clipping needed), trace vs ||Z||^2 rel diff {sci(d_['trace_rel_diff_vs_train_energy'], 2)}"
          + (f", Gram vs partial trace of flat Gram rel diff {sci(d_['flat_gram_partial_trace_rel_diff'], 2)}" if 'flat_gram_partial_trace_rel_diff' in d_ else ""))
    A("Subspace stability diagnostics ([MEASURED]; stability_diagnostics.csv, Fig 6).  'projector distance' = ||QQ^T - Q_altQ_alt^T||_F (maximum possible sqrt(2k)); boundary gap = (lambda_k - lambda_{k+1})/lambda_k.")
    A(f"   Close-pair criterion: relative eigengap < {CLOSE_GAP_REL:g}.  Pairs found inside any retained set or at any truncation boundary: "
      f"{sum(int(float(r.get('n_close_pairs_within_retained') or 0)) for r in stab)}; boundary-close flags: {sum(1 for r in stab if r.get('boundary_pair_close') == 'True')}.")
    hdr = ["object", "config", "dim", "comparison", "proj. distance", "max angle (deg)", "boundary rel. gap"]
    body = []
    for r in stab:
        if r["comparison"] in ("scipy_vs_numpy_eigh_same_gram", "split_half_train_first_vs_second") and r["config_id"] in main_cfgs:
            body.append([r["object"], r["config_id"], r["dimension"], r["comparison"].replace("_same_gram", "").replace("_train_first_vs_second", ""),
                         sci(r["projector_frobenius_distance"], 2), sci(r["max_principal_angle_deg"], 2), fmt(r["boundary_rel_eigengap"], 4)])
    A(text_table(hdr, body))
    A("   scipy-vs-numpy rows: same Gram, two LAPACK paths -> solver-level numerical stability (distances ~1e-7 or below).")
    pg = [float(r["boundary_rel_eigengap"]) for r in stab if r["object"] == "pca" and r["comparison"] == "split_half_train_first_vs_second" and r["config_id"] in main_cfgs]
    pa = [float(r["max_principal_angle_deg"]) for r in stab if r["object"] == "pca" and r["comparison"] == "split_half_train_first_vs_second" and r["config_id"] in main_cfgs]
    A("   split-half rows: Gram from the first vs. second chronological half of TRAIN -> DATA sensitivity of the fitted subspace (a diagnostic, not a test).")
    if pg:
        A(f"   For PCA the boundary relative eigengaps are only {min(pg):.2%}-{max(pg):.2%} and the largest principal angle between the half-fitted subspaces is {min(pa):.1f}-{max(pa):.1f} deg:")
        A("   some boundary directions are poorly determined by the data (the subspace is not stable under data perturbation) even though adjacent boundary directions carry almost equal energy (small eigengap), so swapping them changes train reconstruction error very little.")
    fb = meta.get("fallback_crosscheck")
    if fb:
        A(f"Approximate-fallback cross-check at k={fb['dimension']} (randomized subspace iteration, seed {fb['seed']}, {fb['n_iter']} iterations, oversample {fb['oversample']}; additional diagnostic, NOT used for any result):")
        A(f"   projector distance to exact {sci(fb['projector_frobenius_distance'], 3)}, max principal angle {fb['max_principal_angle_deg']:.2f} deg, max relative eigenvalue error {sci(fb['max_rel_eigenvalue_error'], 3)},")
        A(f"   max relative Ritz residual {sci(fb['max_rel_ritz_residual'], 3)}, train SSE excess over the exact optimum {sci(fb['train_sse_rel_excess'], 3)} (relative).")
        A("   => the fallback reaches near-optimal reconstruction error with a visibly different (not converged) basis; this is why the fallback is labeled approximate and its optimality inequality is not enforced.")
    nclose = sum(int(float(r.get('n_close_pairs_within_retained') or 0)) for r in stab) + sum(1 for r in stab if r.get('boundary_pair_close') == 'True')
    A("Eigenvector sign normalization does not guarantee identical coordinates under (near-)repeated eigenvalues; "
      + ("no such pair was found at the evaluated truncation points under the declared criterion, " if nclose == 0 else "near-degenerate pairs WERE flagged (see CSV); ")
      + "but the split-half diagnostics above show the boundary subspaces can still be data-sensitive.  SHA-256 hashes of feature files are recorded for audit within this software environment only.")
    A("")
    A("11. INTERPRETATION AND CAVEATS")
    A("-" * 110)
    A("(a) Overlapping windows.  Consecutive 24-hour windows share 23 of 24 hours, so samples are strongly dependent: the effective sample size is much smaller than N, errors/metrics are weighted by repeated hours, and")
    A("    no standard errors or confidence intervals are reported (they would need a block bootstrap, not done here).  Differences of a few thousandths in RRE should not be over-interpreted.")
    A("(b) Units.  Reconstruction errors are in Phase-2 standardized units (per station-feature train z-scores); they are not ug/m^3 and not directly comparable across channels' physical importance.  Channels with")
    A("    duplicated meteorology (Section 2) carry multiplicity weight in every Frobenius quantity.")
    held = [(s_, cid) for s_ in ("validation", "test") for cid in main_cfgs]
    held_pca = sum(float(get(s_, "pca", c_)["relative_reconstruction_error"]) < float(get(s_, "hosvd", c_)["relative_reconstruction_error"]) for s_, c_ in held)
    A("(c) Scope of optimality.  Exact PCA's optimality (T4) holds only for TRAIN reconstruction of the flattened centered training matrix.  On held-out splits PCA had the lower RRE in "
      f"{held_pca} of {len(held)} (validation/test x config) comparisons [MEASURED]; that is an empirical observation, and the held-out PCA-minus-HoSVD gap is not guaranteed by T4.  The HoSVD bound (T3) uses TRAIN spectra and is not a held-out guarantee.")
    A("(d) Train-fitted 4-way HoSVD with an uncompressed sample mode is equivalent to a Tucker-type model with identity sample factor; shared factors are fixed after training, so it is a frozen linear map for new windows.")
    A("(e) HoSVD is a NON-optimal truncation (T3 only bounds it): successive/simultaneous mode projection is not the best multilinear rank-(r_t,r_s,r_f) approximation; no HOOI/ALS refinement was run (out of scope).")
    A("(f) Lower reconstruction error is NOT evidence of better next-state classification.  Phase 5 owns accuracy, macro-F1, state-change recall, entry-into-high-state recall, calibrated probabilities, and paired comparison")
    A("    against the Phase 3 Markov persistence baseline (~90.75% Scope-A test accuracy, 0 recall on true regime changes -- a figure quoted from the brief, not recomputed here).")
    A("(g) Not done: any supervised model, kernel/RFF, Markov fusion, threshold tuning, or selection on test; no literature novelty claim is made.")
    A("")
    A("12. TESTS  [MEASURED]")
    A("-" * 110)
    A(pytest_txt.strip())
    A("   (A test suite is evidence, not a formal proof.  Skipped tests, if any, are listed with reasons above.)")
    A("")
    A("13. FILE LIST AND REGENERATION")
    A("-" * 110)
    for p_ in sorted(out.iterdir()):
        if p_.is_file() and p_.name != "Air_Pollution_Phase4_Results.zip":
            A(f"   {p_.name:50s} {p_.stat().st_size:>12d} B")
    A("   figures/: " + ", ".join(sorted(x.name for x in (out / "figures").glob("*.png"))))
    A("   (feature NPZ hashes: phase4_feature_manifest.csv; every row metadata_alignment_passed=" + str(all(m["metadata_alignment_passed"] == "True" for m in man)) + ")")
    A("Regenerate everything from the Phase 2 ZIP (run from the folder that contains phase4_output/):")
    A("   pip install -r phase4_output/requirements.txt")
    A("   python phase4_output/phase4_pca_hosvd.py --phase2-zip Air_Pollution_Phase2_Processed_Data.zip --output phase4_output --seed 42 --configs A,B,C,B_alt1,B_alt2")
    A("   PHASE2_DIR=<extracted phase2_output> python -m pytest -rs -v phase4_output/tests | tee phase4_output/pytest_results.txt")
    A("   python phase4_output/phase4_pca_hosvd.py --output phase4_output --finalize     # rewrites this report, README and the results ZIP")
    A("")
    A("14. IMPLEMENTATION-STATUS SUMMARY")
    A("-" * 110)
    A(f"Completed: {', '.join(cfgs)}.  Not run: {', '.join(c['config_id'] for c in notrun) if notrun else 'none'}.  PCA solver: {meta['pca_solver']['used']}.")
    A("Blockers: none for Phase 4 execution.  This report and all artifacts await independent audit before Phase 5.")
    (out / "PHASE4_RESULTS_AND_MATH.txt").write_text("\n".join(L) + "\n", encoding="utf-8")

    readme = f"""# Phase 4 -- PCA/SVD vs shared-basis HoSVD (unsupervised reconstruction)

Status: executed. Completed configurations: {', '.join(cfgs)}; not run: {', '.join(c['config_id'] for c in notrun) if notrun else 'none'}. PCA solver: {meta['pca_solver']['used']}.
Read `PHASE4_RESULTS_AND_MATH.txt` first (math, measured results, caveats, test results).

## Reproduce
```
pip install -r phase4_output/requirements.txt
python phase4_output/phase4_pca_hosvd.py --phase2-zip Air_Pollution_Phase2_Processed_Data.zip --output phase4_output --seed 42 --configs A,B,C,B_alt1,B_alt2
PHASE2_DIR=<extracted phase2_output> python -m pytest -rs -v phase4_output/tests | tee phase4_output/pytest_results.txt
python phase4_output/phase4_pca_hosvd.py --output phase4_output --finalize
```
Other options: `--phase2-dir`, `--configs`, `--batch-size`, `--pca-solver auto|exact-gram|truncated-fallback`, `--max-memory-gb`, `--fallback-max-k`, `--timing-repeats`, `--no-fallback-crosscheck`.
`--pca-solver exact-gram` never falls back silently; `truncated-fallback` is labeled approximate and restricted to k<=`--fallback-max-k`.

## Key files
- `pca_hosvd_comparison.csv`, `compression_storage.csv`, `hosvd_truncation_bound.csv`, `pca_components_spectrum.csv`, `hosvd_mode_spectra.csv`, `phase4_feature_manifest.csv`, `stability_diagnostics.csv`, `phase4_run_status.csv`
- `features_{{train,validation,test}}_{{pca,hosvd}}_{{A,B,C}}.npz`: keys exactly F (N,k) float32, y int8, target_timestamp datetime64[ns], current_state int8, current_pm25 float32 (NaNs kept). `current_pm25` is metadata, NOT part of F.
- `fitted_center_and_factors.npz` (M, full and per-config HoSVD factors, spectra), `fitted_pca_basis_{{48,192,576}}.npz` (float64 V_k)
- `phase4_metadata.json`, `execution.log`, `figures/`

## Conventions
Features are in Phase-2 standardized units. Flatten order is C (time, station, feature). HoSVD feature index = p*(r_s*r_f)+q*r_f+r. Model parameters are stored and counted as float64; codes are float32.
Scope: unsupervised representation only (no SVM/RFF/Markov/classification). Phase 5 must decide any predictive use; lower reconstruction error does not imply better classification.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    (out / "requirements.txt").write_text(
        f"numpy>=2.0   # tested with {meta['environment']['numpy']}\nscipy>=1.11  # tested with {meta['environment']['scipy']}\n"
        f"matplotlib>=3.7  # tested with {meta['environment'].get('matplotlib')}\npsutil>=5.9  # tested with {meta['environment'].get('psutil')}\n"
        f"pytest>=7.0  # tested with {meta['environment'].get('pytest')}\n", encoding="utf-8")


def finalize(args):
    out = Path(args.output).resolve()
    setup_logging(out, append=True)
    LOG.info("[finalize] writing PHASE4_RESULTS_AND_MATH.txt and README.md, then packaging ZIP")
    write_report(out, toy_pca_2x2(), toy_hosvd_2x2x2())
    zip_outputs(out)


def zip_outputs(out):
    out = Path(out)
    zp = out / "Air_Pollution_Phase4_Results.zip"
    if zp.exists():
        zp.unlink()
    n = 0
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(out.rglob("*")):
            if p.is_dir() or p == zp or "__pycache__" in p.parts or ".pytest_cache" in p.parts or p.suffix == ".pyc":
                continue
            z.write(p, Path("phase4_output") / p.relative_to(out))
            n += 1
    LOG.info(f"[package] {zp} ({zp.stat().st_size} bytes, {n} files)")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Phase 4: matched-dimension PCA/SVD vs shared-basis HoSVD (unsupervised).")
    ap.add_argument("--phase2-zip", default=None)
    ap.add_argument("--phase2-dir", default=None)
    ap.add_argument("--output", default="phase4_output")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--configs", default="A,B,C", help="comma list from A,B,C,B_alt1,B_alt2 (B_alt* are exploratory)")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--pca-solver", choices=["auto", "exact-gram", "truncated-fallback"], default="auto")
    ap.add_argument("--max-memory-gb", type=float, default=None, help="memory budget used by --pca-solver auto")
    ap.add_argument("--fallback-max-k", type=int, default=192, help="largest k run when the approximate fallback is in use")
    ap.add_argument("--timing-repeats", type=int, default=3)
    ap.add_argument("--no-fallback-crosscheck", dest="fallback_crosscheck", action="store_false")
    ap.add_argument("--skip-count-check", action="store_true", help="testing only: do not enforce the exact Phase-2 sample counts")
    ap.add_argument("--finalize", action="store_true", help="only (re)write report/README and the results ZIP from existing outputs")
    return ap.parse_args(argv)


if __name__ == "__main__":
    a = parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    if a.finalize:
        finalize(a)
    else:
        try:
            run_experiment(a)
        except InputContractError:
            sys.exit(2)
