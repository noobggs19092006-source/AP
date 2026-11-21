"""
test_transition_metrics.py – Verify exact synthetic cases.
"""
import pytest
import numpy as np
from pathlib import Path

# Import compute_metrics function
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "source"))
from evaluation import compute_metrics

def test_change_recall_vs_A_change():
    """Current Low -> True High -> Predicted Moderate:
    - Binary change: detected (pred != current)
    - Exact destination: WRONG (pred != true)
    => Change recall counts it, A_change does NOT.
    """
    y_true = np.array([2])        # True: High
    y_pred = np.array([1])        # Predicted: Moderate
    current_state = np.array([0]) # Current: Low
    metrics = compute_metrics(y_true, y_pred, current_state)
    assert metrics["change_TP"] == 1, "Binary change should be detected (TP)"
    assert metrics["change_FP"] == 0
    assert metrics["change_FN"] == 0
    assert metrics["A_change_numerator"] == 0, "Exact destination wrong -> A_change=0"
    assert metrics["A_change_denominator"] == 1

def test_stable_correct():
    """Current Moderate -> True Moderate -> Predicted Moderate"""
    y_true = np.array([1])
    y_pred = np.array([1])
    current_state = np.array([1])
    m = compute_metrics(y_true, y_pred, current_state)
    assert m["change_TN"] == 1
    assert m["change_FP"] == 0
    assert m["A_stable_numerator"] == 1
    assert m["A_stable_denominator"] == 1
    assert m["A_stable"] == 1.0

def test_change_false_alarm():
    """Current Low -> True Low -> Predicted Moderate"""
    y_true = np.array([0])
    y_pred = np.array([1])
    current_state = np.array([0])
    m = compute_metrics(y_true, y_pred, current_state)
    assert m["change_FP"] == 1
    assert m["A_stable_numerator"] == 0
    assert m["A_stable_denominator"] == 1

def test_high_entry_correct():
    """Current Moderate -> True High -> Predicted High"""
    y_true = np.array([2])
    y_pred = np.array([2])
    current_state = np.array([1])
    m = compute_metrics(y_true, y_pred, current_state)
    assert m["high_TP"] == 1
    assert m["high_precision"] == 1.0
    assert m["high_recall"] == 1.0
    assert m["high_F1"] == 1.0

def test_high_entry_miss():
    """Current Low -> True High -> Predicted Low"""
    y_true = np.array([2])
    y_pred = np.array([0])
    current_state = np.array([0])
    m = compute_metrics(y_true, y_pred, current_state)
    assert m["high_FN"] == 1
    assert m["high_TP"] == 0

def test_zero_denominator():
    """Scope A empty (current_state all -1)"""
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    current_state = np.array([-1, -1])
    m = compute_metrics(y_true, y_pred, current_state)
    assert m["scopeA_N"] == 0
    assert m["change_precision"] == 0.0
    assert m["A_change"] == 0.0
    assert m["A_stable"] == 0.0