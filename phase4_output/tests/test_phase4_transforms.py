"""
Test suite for phase4_pca_hosvd.py.  A test suite is evidence, not a formal proof of correctness.

Groups A..M follow the Phase 4 brief.  Tests that need the real Phase 2 arrays or the generated outputs
SKIP (with an explicit reason) when those are not present:
    PHASE2_DIR=/path/to/phase2_output   (directory with train/validation/test_windows.npz)
Run:  pytest -rs -v phase4_output/tests
"""
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
sys.path.insert(0, str(OUT))
import phase4_pca_hosvd as p4  # noqa: E402

SHAPE = (6, 4, 3)           # small synthetic tensor shape (D = 72)
D_SYN = int(np.prod(SHAPE))


# ------------------------------------------------------------------ fixtures / helpers
def make_synthetic(n, seed, shape=SHAPE, ndist=0.3):
    """Structured float32 data: low multilinear rank signal + noise (generic, no repeated eigenvalues)."""
    rng = np.random.default_rng(seed)
    t, s, f = shape
    Ut, Us, Uf = rng.standard_normal((t, 3)), rng.standard_normal((s, 2)), rng.standard_normal((f, 2))
    core = rng.standard_normal((n, 3, 2, 2)) * np.array([3.0, 2.0, 1.0])[None, :, None, None]
    sig = np.einsum("ipqr,ap,bq,cr->iabc", core, Ut, Us, Uf)
    X = sig + ndist * rng.standard_normal((n,) + shape) + 0.5 * rng.standard_normal(shape)[None]
    return X.astype(np.float32)


def fit_models_from_train(Xtr, ranks, batch=64):
    """Mini re-implementation of the fitting flow using ONLY the module's public primitives and ONLY X_train."""
    M, _ = p4.fit_mean(Xtr, batch)
    acc = p4.accumulate_grams(Xtr, M, batch, want_flat=True)
    modes = {}
    for nm, G in zip(["time", "station", "feature"], [acc["Gt"], acc["Gs"], acc["Gf"]]):
        e = p4.eig_desc(G, nm, log=False)
        e["V"] = p4.canonicalize_signs(e["V"])
        modes[nm] = e
    ep = p4.eig_desc(acc["G"], "flat", log=False)
    V = p4.canonicalize_signs(ep["V"])
    k = int(np.prod(ranks))
    Q = [modes[nm]["V"][:, :r] for nm, r in zip(["time", "station", "feature"], ranks)]
    return {"M": M, "acc": acc, "modes": modes, "pca_eig": ep, "V": V,
            "pca": p4.PCAModel(M, V[:, :k]), "hosvd": p4.HoSVDModel(M, *Q), "k": k}


@pytest.fixture(scope="module")
def syn():
    Xtr, Xva, Xte = make_synthetic(300, 1), make_synthetic(80, 2), make_synthetic(90, 3)
    return {"train": Xtr, "val": Xva, "test": Xte, "fit": fit_models_from_train(Xtr, (3, 2, 2))}


def find_phase2_dir():
    cands = [os.environ.get("PHASE2_DIR"), OUT.parent / "phase2_output", OUT / "phase2_output",
             Path.cwd() / "phase2_output"]
    for c in cands:
        if c and (Path(c) / "train_windows.npz").exists():
            return Path(c)
    return None


@pytest.fixture(scope="module")
def real():
    p = find_phase2_dir()
    if p is None:
        pytest.skip("real Phase 2 arrays not found (set PHASE2_DIR)")
    out = {}
    for s in p4.SPLITS:
        keys, d = p4.load_split(p / f"{s}_windows.npz")
        out[s] = (keys, d)
    return out


def need_outputs(*names):
    for n in names:
        if not (OUT / n).exists():
            pytest.skip(f"generated output {n} not present (run the pipeline first)")


def rows(name):
    need_outputs(name)
    with open(OUT / name, newline="") as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------------ A. input contract
def make_split_dict(n=10, shape=(24, 12, 11)):
    ts = (np.datetime64("2016-01-01T00:00:00", "ns") + np.arange(n) * np.timedelta64(1, "h")).astype("datetime64[ns]")
    return {"X": np.zeros((n,) + shape, np.float32), "y": np.zeros(n, np.int8), "target_timestamp": ts,
            "current_state": np.zeros(n, np.int8), "current_pm25": np.full(n, 10.0, np.float32)}


def test_A_contract_accepts_valid_and_reports_unknown_states():
    d = make_split_dict()
    d["current_state"][2] = -1
    d["current_pm25"][2] = np.nan
    rep = p4.validate_split("train", list(d), d)
    assert rep["N"] == 10 and rep["n_current_state_unknown"] == 1 and rep["nan_iff_unknown_state"]


@pytest.mark.parametrize("mutate,msg", [
    (lambda d: d.__setitem__("X", d["X"].astype(np.float64)), "dtype"),
    (lambda d: d["X"].__setitem__((0, 0, 0, 0), np.nan), "non-finite"),
    (lambda d: d.__setitem__("y", np.full(10, 5, np.int8)), "y outside"),
    (lambda d: d["target_timestamp"].__setitem__(3, d["target_timestamp"][2]), "monotone"),
    (lambda d: d.__setitem__("current_state", d["current_state"][:5]), "shape"),
    (lambda d: d.__setitem__("extra", np.zeros(1)), "keys"),
])
def test_A_contract_rejects_violations(mutate, msg):
    d = make_split_dict()
    mutate(d)
    with pytest.raises(p4.InputContractError, match=msg):
        p4.validate_split("train", list(d), d)


def test_A_cross_split_overlap_and_order():
    a, b = make_split_dict(), make_split_dict()
    with pytest.raises(p4.InputContractError):
        p4.validate_cross_split({"train": a, "validation": b})
    b["target_timestamp"] = b["target_timestamp"] + np.timedelta64(100, "h")
    assert p4.validate_cross_split({"train": a, "validation": b})["cross_split_chronological"]


def test_A_real_inputs_counts_and_no_filtering(real):
    expected_unknown = {"train": 132, "validation": 42, "test": 35}
    for s in p4.SPLITS:
        keys, d = real[s]
        rep = p4.validate_split(s, keys, d, p4.EXPECTED_N[s])
        assert rep["N"] == p4.EXPECTED_N[s]
        assert rep["n_current_state_unknown"] == expected_unknown[s]          # unknown states kept, not filtered
        assert rep["logical_float32_bytes"] == {"train": 305496576, "validation": 53716608, "test": 72483840}[s]
    p4.validate_cross_split({s: real[s][1] for s in p4.SPLITS})


# ------------------------------------------------------------------ B. train-only center / freezing
def test_B_center_is_train_only_and_independent_of_other_splits(syn):
    M_ref = syn["train"].astype(np.float64).mean(0)
    assert np.allclose(syn["fit"]["M"], M_ref, rtol=0, atol=1e-12)
    # re-fit with wildly different "validation/test" arrays existing in scope: fit functions never see them
    Xva_bad = syn["val"] * 100.0 + 7
    f2 = fit_models_from_train(syn["train"], (3, 2, 2))
    assert np.array_equal(f2["pca"].V, syn["fit"]["pca"].V)
    assert np.array_equal(f2["hosvd"].Qt, syn["fit"]["hosvd"].Qt)
    assert not np.allclose(Xva_bad.astype(np.float64).mean(0), syn["fit"]["M"])


def test_B_freeze_and_reload_reproduces_codes(syn, tmp_path):
    h, p = syn["fit"]["hosvd"], syn["fit"]["pca"]
    np.savez(tmp_path / "m.npz", M=h.M, Qt=h.Qt, Qs=h.Qs, Qf=h.Qf, V=p.V)
    z = np.load(tmp_path / "m.npz")
    h2, p2 = p4.HoSVDModel(z["M"], z["Qt"], z["Qs"], z["Qf"]), p4.PCAModel(z["M"], z["V"])
    for X in (syn["val"], syn["test"]):
        assert np.array_equal(h.encode(X), h2.encode(X))
        assert np.array_equal(p.encode(X), p2.encode(X))


def test_B_real_center_matches_train_mean():
    need_outputs("fitted_center_and_factors.npz")
    p = find_phase2_dir()
    if p is None:
        pytest.skip("real Phase 2 arrays not found")
    Xtr = np.load(p / "train_windows.npz")["X"]
    M, _ = p4.fit_mean(Xtr, 2048)
    z = np.load(OUT / "fitted_center_and_factors.npz")
    assert np.allclose(z["M"], M, rtol=0, atol=1e-12) and z["M"].dtype == np.float64


# ------------------------------------------------------------------ C. orthonormality / shapes / finiteness
def test_C_orthonormality_shapes_finite(syn):
    f = syn["fit"]
    assert f["pca"].V.shape == (D_SYN, f["k"])
    assert p4.orthonormality_error(f["pca"].V) < p4.ORTHO_ATOL
    for Q, n, r in zip([f["hosvd"].Qt, f["hosvd"].Qs, f["hosvd"].Qf], SHAPE, (3, 2, 2)):
        assert Q.shape == (n, r) and p4.orthonormality_error(Q) < p4.ORTHO_ATOL
    for m in (f["pca"], f["hosvd"]):
        C = m.encode(syn["test"])
        assert C.shape == (len(syn["test"]), f["k"]) and np.isfinite(C).all()
        assert np.isfinite(m.decode(C)).all()


def test_C_real_output_bases_orthonormal():
    need_outputs("fitted_center_and_factors.npz")
    z = np.load(OUT / "fitted_center_and_factors.npz")
    for cid in [str(c) for c in z["config_ids"]]:
        for nm in ("Q_time", "Q_station", "Q_feature"):
            Q = z[f"{cid}_{nm}"]
            assert Q.dtype == np.float64 and p4.orthonormality_error(Q) <= p4.ORTHO_ATOL
            assert p4.canonical_pivot_nonnegative(Q)
    for f in sorted(OUT.glob("fitted_pca_basis_*.npz")):
        V = np.load(f)["V"]
        assert V.dtype == np.float64 and p4.orthonormality_error(V) <= p4.ORTHO_ATOL and p4.canonical_pivot_nonnegative(V)


# ------------------------------------------------------------------ D. sign canonicalization / degeneracy
def test_D_sign_canonicalization_and_invariance():
    rng = np.random.default_rng(0)
    V = rng.standard_normal((20, 7))
    Vc = p4.canonicalize_signs(V)
    assert p4.canonical_pivot_nonnegative(Vc)
    flips = rng.choice([-1.0, 1.0], size=7)
    assert np.array_equal(p4.canonicalize_signs(V * flips[None, :]), Vc)        # sign-invariance
    # spec formula, column by column
    for j in range(7):
        v = V[:, j].copy()
        piv = np.argmax(np.abs(v))
        if v[piv] < 0:
            v *= -1
        assert np.array_equal(v, Vc[:, j])


def test_D_eigenvector_sign_invariance_of_coordinates_projection(syn):
    f = syn["fit"]
    V = f["pca"].V
    flips = np.where(np.arange(V.shape[1]) % 2 == 0, -1.0, 1.0)
    V_neg = V * flips[None, :]
    assert np.array_equal(p4.canonicalize_signs(V_neg), V)
    # un-canonicalized negated basis gives different codes but the identical reconstruction
    Z = f["pca"].center(syn["test"])
    C1, C2 = Z.reshape(len(Z), -1) @ V, Z.reshape(len(Z), -1) @ V_neg
    assert not np.allclose(C1, C2)
    assert np.allclose(C1 @ V.T, C2 @ V_neg.T, atol=1e-10)


def test_D_close_eigenvalue_diagnostic_and_projector_stability():
    rng = np.random.default_rng(1)
    Q, _ = np.linalg.qr(rng.standard_normal((12, 12)))
    lam = np.array([10, 8, 8.0000000001, 5, 3, 2, 1, .5, .2, .1, .05, .01])   # (2,3) nearly repeated
    G = (Q * lam) @ Q.T
    e = p4.eig_desc(G, "toy", log=False)
    gd_inside = p4.eigengap_diagnostics(e["w"], 4)       # pair (2,3) inside retained set
    assert gd_inside["n_close_pairs_within_retained"] == 1 and not gd_inside["boundary_pair_close"]
    gd_boundary = p4.eigengap_diagnostics(e["w"], 2)     # truncation splits the near-degenerate pair
    assert gd_boundary["boundary_pair_close"]
    # projector distances: complete degenerate eigenspace is stable; cutting it is not
    V = e["V"]
    rot = np.eye(12)
    th = 0.7
    rot[1:3, 1:3] = [[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]]
    V_alt = V @ rot                                       # another valid eigenbasis for (near-)degenerate pair
    assert p4.projector_distance(V[:, :3], V_alt[:, :3]) < 1e-8
    assert p4.projector_distance(V[:, :2], V_alt[:, :2]) > 0.3
    # identity used in the implementation equals the explicit Frobenius norm
    A, B = V[:, :2], V_alt[:, :2]
    assert abs(p4.projector_distance(A, B) - np.linalg.norm(A @ A.T - B @ B.T)) < 1e-10


# ------------------------------------------------------------------ E. forward / inverse, einsum vs direct, order
def test_E_full_rank_roundtrip_is_exact(syn):
    f = syn["fit"]
    full_h = p4.HoSVDModel(f["M"], f["modes"]["time"]["V"], f["modes"]["station"]["V"], f["modes"]["feature"]["V"])
    full_p = p4.PCAModel(f["M"], f["V"])
    X = syn["test"].astype(np.float64)
    for m in (full_h, full_p):
        Xh = m.decode(m.encode(syn["test"]))
        assert np.max(np.abs(Xh - X)) < 1e-9


def test_E_einsum_vs_direct_kronecker_and_core_order(syn):
    h = syn["fit"]["hosvd"]
    Z = h.center(syn["val"])
    C = h.encode_centered(Z)
    direct = Z.reshape(len(Z), -1) @ h.kron_basis()
    assert np.allclose(C, direct, atol=1e-11)
    assert np.allclose(h.decode_centered(C).reshape(len(Z), -1), direct @ h.kron_basis().T, atol=1e-11)
    # deterministic C-order: F index = p*(rs*rf) + q*rf + r  -- verify with unit selector factors
    Qt, Qs, Qf = np.eye(6)[:, :3], np.eye(4)[:, :2], np.eye(3)[:, :2]
    m = p4.HoSVDModel(np.zeros(SHAPE), Qt, Qs, Qf)
    Zt = np.zeros((1,) + SHAPE)
    Zt[0, 2, 1, 1] = 5.0                        # p=2, q=1, r=1
    C1 = m.encode_centered(Zt)
    assert np.argmax(np.abs(C1[0])) == 2 * (2 * 2) + 1 * 2 + 1 and C1[0, 2 * 4 + 1 * 2 + 1] == 5.0
    assert np.array_equal(m.encode_centered(Zt), m.encode_centered(Zt))   # deterministic


def test_E_subset_direct_checks(syn):
    for m in (syn["fit"]["pca"], syn["fit"]["hosvd"]):
        chk = p4.subset_direct_check(m, syn["test"], np.arange(0, 60, 3))
        assert chk["max_abs_recon_diff"] < 1e-10 and chk["max_abs_code_diff"] < 1e-10 and chk["shortcut_rel_gap"] < 1e-12


def test_E_real_minibatch_forward_inverse(real):
    """Real Phase 2 rows with real-shape (24,12,11) factors fitted on a TRAIN SUBSET (fast)."""
    X = real["train"][1]["X"]
    sub = X[:1500]
    M, _ = p4.fit_mean(sub, 500)
    acc = p4.accumulate_grams(sub, M, 500, want_flat=False)
    Q = [p4.canonicalize_signs(p4.eig_desc(G, "m", log=False)["V"])[:, :r]
         for G, r in zip([acc["Gt"], acc["Gs"], acc["Gf"]], (4, 3, 4))]
    h = p4.HoSVDModel(M, *Q)
    Xb = X[2000:2064]
    Z = h.center(Xb)
    C = h.encode_centered(Z)
    assert np.allclose(C, Z.reshape(len(Z), -1) @ h.kron_basis(), atol=1e-9)
    Zh = h.decode_centered(C)
    assert np.isclose(np.sum((Z - Zh) ** 2), np.sum(Z ** 2) - np.sum(C ** 2), rtol=1e-10)


# ------------------------------------------------------------------ F. exact-PCA optimality vs HoSVD (train)
@pytest.mark.parametrize("ranks", [(1, 1, 1), (2, 1, 2), (3, 2, 2), (4, 3, 3), (6, 4, 3)])
def test_F_pca_train_sse_le_hosvd(syn, ranks):
    Xtr = syn["train"]
    f = fit_models_from_train(Xtr, ranks)
    tol = p4.ENERGY_REL_TOL * f["acc"]["ssq"]
    rp, _ = p4.evaluate_split(f["pca"], Xtr, 64)
    rh, _ = p4.evaluate_split(f["hosvd"], Xtr, 64)
    assert rp["sse"] <= rh["sse"] + tol
    # exact PCA SSE equals the sum of discarded eigenvalues
    tail = float(p4.tail_energies(f["pca_eig"]["w"])[f["k"]])
    assert abs(rp["sse"] - tail) <= tol


def test_F_real_output_optimality_flags():
    for r in rows("hosvd_truncation_bound.csv"):
        if r.get("pca_le_hosvd_train_sse", "") != "":
            assert r["pca_le_hosvd_train_sse"] == "True" and r["pca_optimality_enforced"] == "True"
            assert float(r["pca_train_sse"]) <= float(r["empirical_train_sse"]) + float(r["numerical_tolerance"])


# ------------------------------------------------------------------ G. bound, identity, eigenvalue care
@pytest.mark.parametrize("ranks", [(1, 1, 1), (2, 2, 1), (3, 2, 2), (4, 3, 2), (5, 3, 3)])
def test_G_hosvd_train_bound_identity_and_three_term_decomposition(syn, ranks):
    Xtr = syn["train"]
    f = fit_models_from_train(Xtr, ranks)
    h = f["hosvd"]
    res, _ = p4.evaluate_split(h, Xtr, 100)
    tails = [float(p4.tail_energies(f["modes"][nm]["w"])[r]) for nm, r in zip(["time", "station", "feature"], ranks)]
    tol = p4.ENERGY_REL_TOL * f["acc"]["ssq"]
    assert res["sse"] <= sum(tails) + tol
    assert abs(res["sse"] - res["sse_identity"]) <= p4.IDENTITY_REL_TOL * res["znorm2"]
    # exact orthogonal 3-term decomposition: SSE = |(I-Pt)Z|^2 + |Pt(I-Ps)Z|^2 + |PtPs(I-Pf)Z|^2
    Z = h.center(Xtr)
    Pt, Ps, Pf = h.Qt @ h.Qt.T, h.Qs @ h.Qs.T, h.Qf @ h.Qf.T
    ap = lambda Z_, P, ax: np.moveaxis(np.tensordot(Z_, P, axes=([ax], [0])), -1, ax)   # noqa: E731
    I_t = Z - ap(Z, Pt, 1)
    t1 = np.sum(I_t ** 2)
    Zt = ap(Z, Pt, 1)
    t2 = np.sum((Zt - ap(Zt, Ps, 2)) ** 2)
    Zts = ap(Zt, Ps, 2)
    t3 = np.sum((Zts - ap(Zts, Pf, 3)) ** 2)
    assert abs(t1 - tails[0]) <= tol                      # first term equals the discarded time-mode tail exactly
    assert abs((t1 + t2 + t3) - res["sse"]) <= tol
    assert t2 <= tails[1] + tol and t3 <= tails[2] + tol  # the other two are bounded by their own tails


def test_G_trace_identities(syn):
    f = syn["fit"]
    zn = f["acc"]["ssq"]
    for nm in ("time", "station", "feature"):
        assert abs(f["modes"][nm]["w"].sum() - zn) <= 1e-10 * zn
    assert abs(f["pca_eig"]["w"].sum() - zn) <= 1e-10 * zn
    G = f["acc"]["G"]
    assert abs(np.trace(G) - zn) <= 1e-10 * zn


def test_G_negative_eigenvalue_policy():
    with pytest.raises(p4.NumericalError):
        p4.eig_desc(np.diag([1.0, 0.5, -0.5]), "bad", log=False)               # significantly negative -> error
    G = np.diag([1e6, 10.0, 1.0, 0.0])
    G[3, 3] = -1e-14                                                            # tiny negative within tolerance
    e = p4.eig_desc(G, "tiny", log=False)
    assert e["n_negative_clipped"] == 1 and e["w"].min() == 0.0 and e["min_raw_eigenvalue"] < 0
    with pytest.raises(p4.NumericalError):
        p4.eig_desc(np.array([[1.0, 0.5], [0.0, 1.0]]), "asym", log=False)


def test_G_real_bound_csv():
    rs = rows("hosvd_truncation_bound.csv")
    assert rs
    for r in rs:
        assert r["bound_passed"] == "True"
        e, b, t = float(r["empirical_train_sse"]), float(r["upper_bound_sse"]), float(r["numerical_tolerance"])
        assert e <= b + t
        assert abs(b - (float(r["tail_time"]) + float(r["tail_station"]) + float(r["tail_feature"]))) <= 1e-6 * b


# ------------------------------------------------------------------ H. RRE / MSE definitions
def test_H_metric_definitions(syn):
    for m in (syn["fit"]["pca"], syn["fit"]["hosvd"]):
        X = syn["test"]
        res, F32 = p4.evaluate_split(m, X, 50)
        Z = m.center(X)
        Zh = m.decode_centered(m.encode_centered(Z))
        sse = np.sum((Z - Zh) ** 2)
        assert math.isclose(res["sse"], sse, rel_tol=1e-12)
        assert math.isclose(res["rre"], math.sqrt(sse / np.sum(Z ** 2)), rel_tol=1e-12)
        assert math.isclose(res["mse"], sse / (len(X) * D_SYN), rel_tol=1e-12)
        assert -1e-12 <= res["retained_energy"] <= 1 + 1e-12
        assert abs(res["rre"] ** 2 + res["retained_energy"] - 1) < 1e-10
        assert F32.dtype == np.float32
        assert abs(res["rre_float32_minus_float64"]) < 1e-6               # float32 export is immaterial here


def test_H_real_csv_identities():
    for r in rows("pca_hosvd_comparison.csv"):
        rre, ret = float(r["relative_reconstruction_error"]), float(r["retained_energy"])
        assert -1e-12 <= ret <= 1 + 1e-12 and abs(rre ** 2 + ret - 1) < 1e-9
        assert math.isclose(float(r["mse"]), float(r["sse"]) / (int(r["n_samples"]) * 3168), rel_tol=1e-12)
        assert math.isclose(rre, math.sqrt(float(r["sse"]) / float(r["znorm2"])), rel_tol=1e-12)
        assert float(r["identity_abs_diff"]) <= p4.IDENTITY_REL_TOL * float(r["znorm2"])
        assert abs(float(r["rre_float32_minus_float64"])) < 1e-9


# ------------------------------------------------------------------ I. feature-file metadata parity
def test_I_feature_file_contract_synthetic(tmp_path):
    n = 12
    d = make_split_dict(n)
    d["current_state"][[1, 5]] = -1
    d["current_pm25"][[1, 5]] = np.nan
    d["y"] = (np.arange(n) % 3).astype(np.int8)
    F = np.random.default_rng(0).standard_normal((n, 7)).astype(np.float32)
    meta = {k: d[k] for k in ["y", "target_timestamp", "current_state", "current_pm25"]}
    path = tmp_path / "f.npz"
    p4.save_features(path, F, meta)
    assert p4.verify_feature_file(path, F, meta, n, 7)
    z = np.load(path)
    assert list(z.files) == ["F", "y", "target_timestamp", "current_state", "current_pm25"]
    assert np.isnan(z["current_pm25"]).sum() == 2 and z["F"].dtype == np.float32
    meta_bad = dict(meta, current_pm25=meta["current_pm25"] + 1)
    assert not p4.verify_feature_file(path, F, meta_bad, n, 7)


def test_I_real_feature_files_match_phase2_metadata(real):
    man = rows("phase4_feature_manifest.csv")
    assert man
    for m in man:
        s = m["split"]
        d = real[s][1]
        with np.load(OUT / m["relative_path"]) as z:
            assert list(z.files) == ["F", "y", "target_timestamp", "current_state", "current_pm25"]
            assert z["F"].shape == (p4.EXPECTED_N[s], int(m["k"])) and z["F"].dtype == np.float32 and np.isfinite(z["F"]).all()
            assert z["y"].dtype == np.int8 and z["current_state"].dtype == np.int8
            assert z["target_timestamp"].dtype == np.dtype("datetime64[ns]") and z["current_pm25"].dtype == np.float32
            assert np.array_equal(z["y"], d["y"]) and np.array_equal(z["current_state"], d["current_state"])
            assert np.array_equal(z["target_timestamp"], d["target_timestamp"])
            assert np.array_equal(z["current_pm25"], d["current_pm25"], equal_nan=True)
            assert z["current_pm25"].tobytes() == d["current_pm25"].tobytes()
            assert (np.diff(z["target_timestamp"].astype("int64")) > 0).all()
        assert m["metadata_alignment_passed"] == "True"
        assert hashlib.sha256((OUT / m["relative_path"]).read_bytes()).hexdigest() == m["sha256"]


def test_I_manifest_covers_every_completed_config_split_method():
    man = rows("phase4_feature_manifest.csv")
    status = {r["config_id"]: r["status"] for r in rows("phase4_run_status.csv")}
    got = {(m["split"], m["method"], m["config_id"]) for m in man}
    for cid, st in status.items():
        if st == "COMPLETED":
            for s in p4.SPLITS:
                assert (s, "hosvd", cid) in got
                if not p4.CONFIGS[cid][1]:
                    assert (s, "pca", cid) in got


# ------------------------------------------------------------------ J. bytes arithmetic / dtypes
def test_J_storage_arithmetic():
    d = 3168
    s = p4.storage_numbers("pca", 48, None, 24108, d)
    assert s["basis_bytes"] == d * 48 * 8 and s["mean_bytes"] == d * 8
    assert s["model_bytes"] == d * 8 + d * 48 * 8 and s["feature_bytes"] == 24108 * 48 * 4
    assert s["raw_logical_bytes"] == 305496576 and s["dimension_ratio"] == 66.0
    assert s["total_estimated_bytes"] == s["feature_bytes"] + s["model_bytes"]
    assert math.isclose(s["effective_storage_ratio"], s["raw_logical_bytes"] / s["total_estimated_bytes"])
    h = p4.storage_numbers("hosvd", 192, (8, 4, 6), 4239, d)
    assert h["basis_bytes"] == (24 * 8 + 12 * 4 + 11 * 6) * 8 == 2448
    assert h["model_parameter_count"] == d + 306 and h["raw_logical_bytes"] == 53716608
    assert h["dimension_ratio"] == 16.5
    assert p4.storage_numbers("pca", 576, None, 5720, d)["dimension_ratio"] == 5.5
    # equal k is not equal storage
    assert p4.storage_numbers("pca", 192, None, 100, d)["model_bytes"] > 100 * p4.storage_numbers("hosvd", 192, (8, 4, 6), 100, d)["model_bytes"]


def test_J_real_storage_csv_matches_dtypes_on_disk():
    st = rows("compression_storage.csv")
    for r in st:
        assert r["model_dtype"] == "float64" and r["feature_dtype"] == "float32"
        k, n = int(r["k"]), int(r["raw_logical_bytes"]) // (3168 * 4)
        assert int(r["feature_bytes"]) == n * k * 4
    need_outputs("fitted_center_and_factors.npz")
    z = np.load(OUT / "fitted_center_and_factors.npz")
    assert z["M"].dtype == np.float64
    for f in OUT.glob("fitted_pca_basis_*.npz"):
        V = np.load(f)["V"]
        k = V.shape[1]
        assert V.dtype == np.float64
        pca_row = [r for r in st if r["method"] == "pca" and int(r["k"]) == k][0]
        assert int(pca_row["basis_bytes"]) == V.nbytes


# ------------------------------------------------------------------ K. no test-driven selection / train-only fits
def test_K_fit_primitives_take_only_train_and_ranks_are_declared():
    import inspect
    for fn in (p4.fit_mean, p4.accumulate_grams, p4.fit_pca_randomized):
        names = list(inspect.signature(fn).parameters)
        assert names[0] == "X" and not any(n in names for n in ("Xva", "Xte", "X_val", "X_test", "y"))
    assert {k: v[0] for k, v in p4.CONFIGS.items()} == {"A": (4, 3, 4), "B": (8, 4, 6), "C": (12, 8, 6),
                                                       "B_alt1": (6, 4, 8), "B_alt2": (8, 6, 4)}
    for cid, (r, _) in p4.CONFIGS.items():
        assert int(np.prod(r)) in (48, 192, 576)


def test_K_labels_never_used_in_fit(syn):
    # decomposition depends only on X: fit twice (identical X) -> bit-identical; labels are not an argument anywhere
    f1 = fit_models_from_train(syn["train"], (3, 2, 2))
    assert np.array_equal(f1["V"][:, :12], syn["fit"]["V"][:, :12])


def test_K_real_run_used_only_declared_configs():
    rs = rows("pca_hosvd_comparison.csv")
    for r in rs:
        exp = p4.CONFIGS[r["config_id"]][0]
        if r["method"] == "hosvd":
            assert (int(r["r_time"]), int(r["r_station"]), int(r["r_feature"])) == exp
        else:
            assert r["r_time"] == "" and r["r_station"] == "" and r["r_feature"] == ""
        assert int(r["k"]) == int(np.prod(exp))


# ------------------------------------------------------------------ L. missing / incomplete C
def test_L_not_run_is_explicit_and_never_fabricated():
    plan = p4.plan_configs(["A", "B", "C"], "truncated-fallback", fallback_max_k=192)
    st = {p["config_id"]: p for p in plan}
    assert st["A"]["run"] and st["B"]["run"]
    assert (not st["C"]["run"]) and st["C"]["status"] == "NOT RUN" and "k=576" in st["C"]["reason"]
    plan2 = p4.plan_configs(["A", "B", "C"], "exact-gram", fallback_max_k=192)
    assert all(p["run"] for p in plan2)
    with pytest.raises(ValueError):
        p4.plan_configs(["Z"], "exact-gram", 192)


def test_L_solver_choice_never_silent():
    assert p4.choose_pca_solver("exact-gram", 24108, 3168, 0.01, 1.0)[0] == "exact-gram"          # forced stays forced
    assert p4.choose_pca_solver("truncated-fallback", 24108, 3168, 100.0, 100.0)[0] == "truncated-fallback"
    s, why = p4.choose_pca_solver("auto", 24108, 3168, 0.05, None)
    assert s == "truncated-fallback" and "auto" in why
    assert p4.choose_pca_solver("auto", 24108, 3168, 8.0, None)[0] == "exact-gram"


def test_L_real_status_and_no_placeholder_rows():
    st = rows("phase4_run_status.csv")
    comp = rows("pca_hosvd_comparison.csv")
    done = {r["config_id"] for r in st if r["status"] == "COMPLETED"}
    for r in comp:
        assert r["status"] == "completed" and r["config_id"] in done
        assert r["relative_reconstruction_error"] not in ("", "nan", "NaN")
    for r in st:
        if r["status"] == "NOT RUN":
            assert r["reason"] and all(x["config_id"] != r["config_id"] for x in comp)


def test_L_randomized_fallback_recovers_top_subspace_and_is_labeled(syn):
    Xtr = syn["train"]
    f = syn["fit"]
    k = 6
    V, th, diag = p4.fit_pca_randomized(Xtr, f["M"], k, 64, seed=7, oversample=10, n_iter=8)
    assert "approximate" in diag["method"] and diag["max_rel_ritz_residual_top_k"] < 1e-3
    assert np.allclose(th, f["pca_eig"]["w"][:k], rtol=1e-4)
    assert p4.projector_distance(p4.canonicalize_signs(V), f["V"][:, :k]) < 0.05


# ------------------------------------------------------------------ M. hand-worked toy examples (ILLUSTRATIVE)
def test_M_toy_pca_2x2():
    t = p4.toy_pca_2x2()
    assert np.allclose(t["gram"], [[5, 4], [4, 5]])
    assert np.allclose(t["eigenvalues"], [9, 1])
    assert np.allclose(t["v1"], np.array([1, 1]) / math.sqrt(2))
    assert np.allclose(t["rank1_recon"], [[1.5, 1.5], [1.5, 1.5]])
    assert math.isclose(t["sse"], 1.0) and math.isclose(t["retained_fraction"], 0.9)


def test_M_toy_hosvd_2x2x2():
    t = p4.toy_hosvd_2x2x2()
    for G in t["grams"]:
        assert np.allclose(G, np.diag([4.0, 1.0]))            # all three unfoldings: singular values 2 and 1
    assert np.allclose(t["core"], 2.0)
    assert math.isclose(t["sse"], 1.0) and math.isclose(t["bound"], 3.0) and t["sse"] <= t["bound"]
    assert np.allclose(t["tails"], [1.0, 1.0, 1.0])
    # explicit unfolding eigenvalues for a generic random tensor: Gram spectra equal squared SVD values of unfoldings
    T = np.random.default_rng(3).standard_normal((2, 2, 2))
    res = p4.full_hosvd_3way(T, (1, 1, 1))
    for mode in range(3):
        sv = np.linalg.svd(np.moveaxis(T, mode, 0).reshape(2, -1), compute_uv=False)
        assert np.allclose(res["eigenvalues"][mode], sv ** 2)
    assert res["sse"] <= res["bound"] + 1e-12


# ------------------------------------------------------------------ extra: end-to-end smoke test on tiny synthetic Phase 2 data
def test_end_to_end_smoke_on_synthetic_phase2_dir(tmp_path):
    rng = np.random.default_rng(5)
    p2 = tmp_path / "phase2_output"
    p2.mkdir()
    start = np.datetime64("2014-01-01T00:00:00", "ns")
    off = 0
    for s, n in [("train", 160), ("validation", 40), ("test", 50)]:
        X = make_synthetic(n, 10 + n, shape=(24, 12, 11)).astype(np.float32)
        ts = (start + (off + np.arange(n)) * np.timedelta64(1, "h")).astype("datetime64[ns]")
        off += n + 24
        cs = rng.integers(0, 3, n).astype(np.int8)
        cs[3] = -1
        pm = rng.uniform(5, 150, n).astype(np.float32)
        pm[3] = np.nan
        np.savez(p2 / f"{s}_windows.npz", X=X, y=rng.integers(0, 3, n).astype(np.int8), target_timestamp=ts,
                 current_state=cs, current_pm25=pm)
    out = tmp_path / "out"
    args = p4.parse_args(["--phase2-dir", str(p2), "--output", str(out), "--configs", "A", "--batch-size", "64",
                          "--skip-count-check", "--no-fallback-crosscheck", "--timing-repeats", "1"])
    p4.run_experiment(args)
    meta = json.load(open(out / "phase4_metadata.json"))
    assert meta["status"] == "COMPLETED" and meta["completed_configs"] == ["A"]
    comp = list(csv.DictReader(open(out / "pca_hosvd_comparison.csv")))
    assert len(comp) == 6 and {r["method"] for r in comp} == {"pca", "hosvd"}
    train = {r["method"]: float(r["sse"]) for r in comp if r["split"] == "train"}
    assert train["pca"] <= train["hosvd"] * (1 + 1e-9)
    b = list(csv.DictReader(open(out / "hosvd_truncation_bound.csv")))[0]
    assert b["bound_passed"] == "True"
    for m in csv.DictReader(open(out / "phase4_feature_manifest.csv")):
        assert m["metadata_alignment_passed"] == "True"
