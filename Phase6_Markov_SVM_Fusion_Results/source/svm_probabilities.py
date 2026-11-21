"""
svm_probabilities.py – Load calibrated SVM/RFF probabilities or flag unsupported.
Returns (N,3) valid probability vectors for class [0,1,2], with is_calibrated flag and reason.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def load_linear_probas(model_path, calibrator_path, scaler_npz_path, features_path, config_id):
    model = pickle.load(open(model_path, "rb"))
    cal = pickle.load(open(calibrator_path, "rb"))
    sc = np.load(scaler_npz_path, allow_pickle=True)
    d = np.load(features_path, allow_pickle=True)
    Fz = (d["F"] - sc["mean"]) / sc["scale"]
    scores = model.decision_function(Fz)
    probs = cal.predict_proba(scores)
    classes = model.classes_
    assert list(classes) == [0, 1, 2], f"Classes {classes!r} differ from expected [0,1,2]"
    return probs, "linear+calibrated"


def load_rff_probas(model_path, calibrator_path, scaler_npz_path, features_path, mapping_npz_path=None):
    m = pickle.load(open(model_path, "rb"))
    cal = pickle.load(open(calibrator_path, "rb"))
    sc = np.load(scaler_npz_path, allow_pickle=True)
    d = np.load(features_path, allow_pickle=True)
    Fz = (d["F"] - sc["mean"]) / sc["scale"]
    # revised dict form
    if isinstance(m, dict):
        z = m["sampler"].transform(Fz)
    else:
        raise ValueError(f"Unrecognised RFF model format at {model_path}")
    scores = m["classifier"].decision_function(z)
    probs = cal.predict_proba(scores)
    assert list(m["classifier"].classes_) == [0, 1, 2]
    return probs, "RFF_revised+calibrated"


def load_rff_unrevised_probas(model_path, calibrator_path, scaler_npz_path, features_path, mapping_npz_path):
    clf = pickle.load(open(model_path, "rb"))
    cal = pickle.load(open(calibrator_path, "rb"))
    sc = np.load(scaler_npz_path, allow_pickle=True)
    mp = np.load(mapping_npz_path, allow_pickle=True)
    d = np.load(features_path, allow_pickle=True)
    Fz = (d["F"] - sc["mean"]) / sc["scale"]
    W = mp["random_weights"]; b = mp["random_offset"]
    z = np.sqrt(2.0 / W.shape[1]) * np.cos(Fz @ W + b)
    scores = clf.decision_function(z)
    probs = cal.predict_proba(scores)
    return probs, "RFF_mapping+calibrated"


def load_exact_rbf_hard(model_path, scaler_npz_path, features_path):
    """Only returns hard predictions; no calibrated probabilities."""
    m = pickle.load(open(model_path, "rb"))
    svc = m["classifier"]
    sc = np.load(scaler_npz_path, allow_pickle=True)
    d = np.load(features_path, allow_pickle=True)
    Fz = (d["F"] - sc["mean"]) / sc["scale"]
    hard = svc.predict(Fz)
    return hard, None, "exact_RBF_no_calibration"
