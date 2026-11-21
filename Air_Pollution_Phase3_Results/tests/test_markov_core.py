"""Unit tests. Everything except the final class runs WITHOUT the real dataset."""
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import markov_core as mc

SEQ = [0, 0, 1, 1, 2, 2, 1, 0, 0]          # L,L,M,M,H,H,M,L,L
C_EXP = np.array([[2, 1, 0], [1, 1, 1], [0, 1, 1]])
F = Fraction


def fr(rows):
    return np.array([[float(F(x)) for x in r] for r in rows])


P_EXP = fr([["2/3", "1/3", "0"], ["1/3", "1/3", "1/3"], ["0", "1/2", "1/2"]])
P2_EXP = fr([["5/9", "1/3", "1/9"], ["1/3", "7/18", "5/18"], ["1/6", "5/12", "5/12"]])
PI_EXP = np.array([3 / 8, 3 / 8, 1 / 4])


def fitted():
    C = mc.counts_from_sequence(SEQ)
    P, _, prior, fb = mc.fit_markov(C, np.array([1, 1, 1]))
    return C, P


# ---------------- hand-worked example
def test_counts_P_P2_pi():
    C, P = fitted()
    assert (C == C_EXP).all() and C.sum() == 8
    assert np.allclose(P, P_EXP, atol=1e-15, rtol=0)
    assert np.allclose(P @ P, P2_EXP, atol=1e-15, rtol=0)
    st = mc.stationary_analysis(P)
    assert np.allclose(st["pi"], PI_EXP, atol=1e-12, rtol=0)
    assert st["stationary_residual_inf_norm"] < 1e-12 and st["constraint_matrix_rank"] == 3


def test_example_spectrum_and_gap():
    _, P = fitted()
    st = mc.stationary_analysis(P)
    ev = np.sort(st["eigenvalues"].real)[::-1]
    assert np.allclose(ev, [1, 0.59359214, -0.09359214], atol=1e-7)
    assert st["irreducible"] and st["aperiodic_given_irreducible"] and st["algebraic_multiplicity_of_1"] == 1
    assert st["spectral_gap"] == pytest.approx(0.40640786, abs=1e-7)


# ---------------- tie-break
def test_tie_break_locked_rule():
    C, _ = fitted()
    modal, ties = mc.modal_rule(C, majority=0)
    assert modal[1] == 1 and ties[1] == [0, 1, 2]       # three-way tie -> current state
    assert modal[2] == 2 and ties[2] == [1, 2]          # two-way tie -> current state
    assert modal[0] == 0
    # current state NOT in tie set -> smallest winning index
    C2 = np.array([[1, 5, 5], [1, 1, 4], [2, 2, 2]])    # row0 tie {1,2}; row1 unique 2
    m2, t2 = mc.modal_rule(C2, majority=2)
    assert m2[0] == 1 and t2[0] == [1, 2]
    assert m2[1] == 2
    assert m2[2] == 2                                   # three-way tie, current=2 in set
    C3 = np.array([[0, 3, 3], [4, 0, 4], [0, 0, 1]])    # row1 tie {0,2}, current 1 not in set -> 0
    m3, _ = mc.modal_rule(C3, majority=0)
    assert m3[1] == 0 and m3[2] == 2


def test_majority_tie_smallest_index():
    assert mc.majority_label(np.array([5, 5, 3])) == 0
    assert mc.majority_label(np.array([1, 7, 7])) == 1


def test_zero_support_row_fallback():
    C = np.array([[3, 1, 0], [0, 0, 0], [0, 2, 2]])
    P, rown, prior, fb = mc.fit_markov(C, np.array([2, 2, 4]))
    assert fb == [1] and np.allclose(P[1], [0.25, 0.25, 0.5])
    assert mc.is_row_stochastic(P)
    modal, ties = mc.modal_rule(C, majority=2)
    assert ties[1] is None and modal[1] == 2


# ---------------- gap handling / timestamps
def test_gap_not_treated_as_one_hour_transition():
    base = int(np.datetime64("2020-01-01T00", "h").astype(np.int64))
    full = np.array([0, 0, 1, 2, 2, 1, 1, 0, 0, 1])           # true states at hours 0..9
    keep = [h for h in range(1, 10) if h != 4]                 # no row for target hour 4 (target missing)
    ts = (np.datetime64("2020-01-01T00", "h") + np.array(keep)).astype("datetime64[ns]")
    y = full[keep]
    cs = full[np.array(keep) - 1]
    # aligned pairs: every pair is a genuine one-hour transition
    C = mc.counts_from_pairs(cs, y)
    true_pairs = [(full[t - 1], full[t]) for t in keep]
    assert C.sum() == len(true_pairs) == 8
    assert all(cs[i] == full[keep[i] - 1] for i in range(len(keep)))
    # the forbidden shortcut would pair hour 3 with hour 5 across the gap
    i3 = keep.index(3)
    assert (y[i3], y[i3 + 1]) == (full[3], full[5]) and keep[i3 + 1] - keep[i3] == 2
    naive = mc.counts_from_pairs(y[:-1], y[1:])
    assert not (naive == C).all()
    # series reconstruction: hour 4 is recovered ONLY through current_state of the row at t=5
    ser = mc.build_state_series(ts, y, cs)
    s = ser["series"]; off = ser["start_hour"] - base
    assert s[4 - off] == full[4] and (s >= 0).all()
    assert len(mc.triples_from_series(s)[0]) == 8              # hours 2..9
    assert len(mc.strict_join_triples(ts, y, cs)[0]) == 6      # t=2,3,6,7,8,9 (t=5 lacks row t-1=4)
    # now make hour 4 genuinely unobserved
    cs2 = cs.copy(); cs2[keep.index(5)] = -1
    ser2 = mc.build_state_series(ts, y, cs2)
    s2 = ser2["series"]; off2 = ser2["start_hour"] - base
    assert s2[4 - off2] == -1
    a2, b2, c2, _ = mc.triples_from_series(s2)
    assert len(a2) == 5                                        # ends at hours 2,3,7,8,9
    assert len(mc.strict_join_triples(ts, y, cs2)[0]) == 5     # t=2,3,7,8,9


def test_series_conflict_detected():
    ts = (np.datetime64("2020-01-01T00", "h") + np.array([1, 2])).astype("datetime64[ns]")
    ser = mc.build_state_series(ts, np.array([0, 1]), np.array([0, 2]))   # y[0]=0 but cs[1]=2
    assert ser["n_conflicts"] == 1


def test_non_whole_hour_rejected():
    with pytest.raises(ValueError):
        mc.hour_index(np.array(["2020-01-01T00:30"], dtype="datetime64[ns]"))


# ---------------- missing state / coverage
def test_missing_state_fallback_and_coverage():
    C, P = fitted()
    modal, _ = mc.modal_rule(C, 1)
    prior = np.array([0.2, 0.5, 0.3]); maj = 1
    cs = np.array([0, -1, 2, -1, 1]); y = np.array([0, 1, 2, 2, 0])
    pm, prm = mc.predict_majority(5, maj, prior)
    pp, prp = mc.predict_persistence(cs, maj, prior)
    pk, prk = mc.predict_markov(cs, P, modal, maj, prior)
    assert (pm == 1).all()
    assert (pp == [0, 1, 2, 1, 1]).all() and (pk == [0, 1, 2, 1, 1]).all()
    assert np.allclose(prk[1], prior) and np.allclose(prk[3], prior) and np.allclose(prk[0], P[0])
    assert np.allclose(prp[0], [1, 0, 0]) and np.allclose(prp[1], prior)
    assert mc.is_row_stochastic(prk) and mc.is_row_stochastic(prm) and mc.is_row_stochastic(prp)
    known = cs >= 0
    assert known.sum() == 3 and (~known).sum() == 2           # Scope A n / fallback n
    # same eligible rows -> identical n for all models
    assert len(y[known]) == 3


def test_invalid_labels_rejected():
    with pytest.raises(ValueError):
        mc.counts_from_pairs(np.array([0, 3]), np.array([0, 1]))
    with pytest.raises(ValueError):
        mc.counts_from_pairs(np.array([0, 1]), np.array([0, -1]))


# ---------------- numerical properties
def test_row_stochastic_checks():
    _, P = fitted()
    assert mc.is_row_stochastic(P) and mc.is_row_stochastic(P @ P)
    bad = P.copy(); bad[0, 0] += 1e-6
    assert not mc.is_row_stochastic(bad)
    bad2 = P.copy(); bad2[0, 0] = -0.1; bad2[0, 1] += 0.1
    assert not mc.is_row_stochastic(bad2)
    assert not mc.is_row_stochastic(np.full((3, 3), np.nan))


def test_reducible_and_periodic_chains():
    red = np.array([[1, 0, 0], [0.5, 0.5, 0], [0, 0.3, 0.7]], dtype=float)
    st = mc.stationary_analysis(red)
    assert not st["irreducible"] and st["spectral_gap"] is None and "not interpreted" in st["spectral_gap_status"]
    two_absorb = np.diag([1.0, 1.0, 1.0])
    st2 = mc.stationary_analysis(two_absorb)
    assert st2["algebraic_multiplicity_of_1"] == 3 and not st2["unique_solution_of_constrained_system"]
    cyc = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
    st3 = mc.stationary_analysis(cyc)
    assert st3["irreducible"] and not st3["aperiodic_given_irreducible"] and st3["spectral_gap"] is None
    assert st3["state_periods"] == [3, 3, 3]
    assert np.iscomplexobj(st3["eigenvalues"]) and np.allclose(st3["pi"], 1 / 3)


def test_expected_run_length():
    P = np.array([[0.75, 0.25, 0], [0, 0.5, 0.5], [0, 0, 1.0]])
    r = mc.expected_run_lengths(P)
    assert r[0] == pytest.approx(4) and r[1] == pytest.approx(2) and r[2] == float("inf")


# ---------------- metrics vs sklearn
def test_metrics_match_sklearn_and_absent_class():
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                                 f1_score, precision_score, recall_score)
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 500); p = np.where(rng.random(500) < 0.7, y, rng.integers(0, 3, 500))
    m = mc.classification_metrics(mc.confusion(y, p))
    kw = dict(labels=[0, 1, 2], average="macro", zero_division=0)
    assert m["accuracy"] == pytest.approx(accuracy_score(y, p))
    assert m["macro_precision"] == pytest.approx(precision_score(y, p, **kw))
    assert m["macro_recall"] == pytest.approx(recall_score(y, p, **kw))
    assert m["macro_f1"] == pytest.approx(f1_score(y, p, **kw))
    assert m["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(y, p))
    assert (mc.confusion(y, p) == confusion_matrix(y, p, labels=[0, 1, 2])).all()
    # class 2 absent from truth: label axes preserved, balanced accuracy flags 2 classes
    y2 = np.array([0, 0, 1, 1]); p2 = np.array([0, 2, 1, 1])
    cm = mc.confusion(y2, p2); assert cm.shape == (3, 3)
    m2 = mc.classification_metrics(cm)
    assert m2["balanced_accuracy_n_classes"] == 2 and m2["balanced_accuracy"] == pytest.approx(0.75)
    assert m2["macro_recall"] == pytest.approx(0.5)             # zero_division=0 for absent class


def test_brier_and_logloss():
    prob = np.array([[1.0, 0, 0], [0.5, 0.5, 0]])
    y = np.array([0, 2])
    assert mc.brier_score(prob, y) == pytest.approx((0 + (0.25 + 0.25 + 1)) / 2)
    ll, nz = mc.log_loss_clipped(prob, y)
    assert nz == 1 and np.isfinite(ll)
    assert mc.log_loss_clipped(np.array([[0.25, 0.25, 0.5]]), np.array([2]))[0] == pytest.approx(-np.log(0.5))


def test_change_metrics_denominators():
    cs = np.array([0, 0, 1, 1, 2, 1]); y = np.array([0, 2, 1, 2, 2, 0]); pred = np.array([0, 2, 2, 1, 2, 0])
    r = mc.change_metrics(y, cs, pred)
    assert r["n_true_change"] == 3 and r["n_pred_change"] == 3 and r["n_change_detected_tp"] == 2
    assert r["change_precision"] == pytest.approx(2 / 3) and r["change_recall"] == pytest.approx(2 / 3)
    assert r["n_enter_high_events"] == 2                       # rows 1 and 3
    assert r["n_enter_high_predicted_correctly"] == 1 and r["enter_high_recall"] == 0.5
    assert r["n_pred_enter_high"] == 2 and r["enter_high_precision"] == 0.5
    assert r["next_state_accuracy_given_change"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        mc.change_metrics(y, np.array([-1, 0, 0, 0, 0, 0]), pred)


# ---------------- chronological isolation
def test_split_isolation():
    b = {"train": ("2013-03-01", "2015-12-31"), "validation": ("2016-01-01", "2016-06-30"),
         "test": ("2016-07-01", "2017-02-28")}
    mk = lambda a: np.array(a, dtype="datetime64[ns]")
    good = {"train": mk(["2013-03-02T00", "2015-12-31T23"]), "validation": mk(["2016-01-02T00", "2016-06-30T23"]),
            "test": mk(["2016-07-02T00", "2017-02-28T23"])}
    mc.check_split_isolation(good, b)
    bad = dict(good); bad["validation"] = mk(["2016-01-01T10", "2016-06-30T23"])      # window reaches into train
    with pytest.raises(ValueError):
        mc.check_split_isolation(bad, b)
    bad = dict(good); bad["train"] = mk(["2013-03-02T00", "2016-01-01T05"])         # target after split end
    with pytest.raises(ValueError):
        mc.check_split_isolation(bad, b)
    bad = dict(good); bad["test"] = mk(["2016-07-02T05", "2016-07-02T04"])          # not increasing
    with pytest.raises(ValueError):
        mc.check_split_isolation(bad, b)


# ---------------- real-output acceptance (skipped if outputs absent)
OUT = Path(__file__).resolve().parents[1] / "phase3_output"


@pytest.mark.skipif(not (OUT / "markov_baseline_metrics.csv").exists(), reason="run phase3_markov.py first")
class TestRealOutputs:
    def test_metrics_table(self):
        m = pd.read_csv(OUT / "markov_baseline_metrics.csv")
        assert len(m) == 18
        a = m[m.scope == "observed_current_only"].groupby("split").n_evaluated.first().to_dict()
        b = m[m.scope == "full_split_with_prior_fallback"].groupby("split").n_evaluated.first().to_dict()
        assert a == {"train": 23976, "validation": 4197, "test": 5685}
        assert b == {"train": 24108, "validation": 4239, "test": 5720}
        fb = m[(m.scope == "full_split_with_prior_fallback") & (m.model != "majority")].groupby("split").n_fallback.first().to_dict()
        assert fb == {"train": 132, "validation": 42, "test": 35}
        assert m[m.model == "majority"].n_fallback.sum() == 0
        assert m[m.scope == "observed_current_only"].n_fallback.sum() == 0
        assert (m[m.scope == "full_split_with_prior_fallback"].prediction_coverage_pct == 100).all()
        assert (m[m.scope == "observed_current_only"].prediction_coverage_pct < 100).all()

    def test_matrices(self):
        C = pd.read_csv(OUT / "markov_transition_counts.csv", index_col=0).to_numpy()
        P = pd.read_csv(OUT / "markov_transition_matrix.csv", index_col=0).to_numpy()
        P2 = pd.read_csv(OUT / "markov_two_step_matrix.csv", index_col=0).to_numpy()
        assert C.sum() == 23976 and mc.is_row_stochastic(P, 1e-9) and mc.is_row_stochastic(P2, 1e-9)
        assert np.allclose(C / C.sum(1, keepdims=True), P, atol=1e-11) and np.allclose(P @ P, P2, atol=1e-11)
        npz = np.load(OUT / "markov_model_parameters.npz")
        assert np.allclose(npz["P"], P, atol=1e-11)

    @pytest.mark.parametrize("split", ["validation", "test"])
    def test_prediction_schema(self, split):
        cols = ["target_timestamp", "y", "current_state", "scope_a_eligible", "pred_majority",
                "pred_persistence_scope_b", "pred_markov_scope_b", "prob_majority_0", "prob_majority_1",
                "prob_majority_2", "prob_markov_0", "prob_markov_1", "prob_markov_2"]
        df = pd.read_csv(OUT / f"markov_predictions_{split}.csv")
        assert list(df.columns) == cols
        assert len(df) == {"validation": 4239, "test": 5720}[split]
        assert pd.to_datetime(df.target_timestamp).is_monotonic_increasing
        assert df[[f"prob_markov_{i}" for i in range(3)]].sum(axis=1).sub(1).abs().max() < 1e-9

    def test_P_hash_unchanged(self):
        meta = json.loads((OUT / "phase3_metadata.json").read_text())
        assert meta["P_unchanged_across_stages"] is True
        assert len(set(meta["P_sha256"].values())) == 1
