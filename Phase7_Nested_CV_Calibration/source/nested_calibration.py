"""
nested_calibration.py – Out-of-sample Platt calibration for Exact RBF SVMs.
Implements nested CV isolation: train RBF on fold-training, calibrate on OOF validation folds.
Uses Per-Class Platt scaling for multiclass probability calibration.
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pathlib import Path


def train_rbf_with_nested_calibration(X_train, y_train, X_val, y_val,
                                       n_components=3, random_state=42, C=1.0):
    """
    Train Exact RBF SVM and calibrate using OOF validation fold.
    Returns (calibrated_proba, calibration_info).

    This implements proper nested CV: the validation fold is NEVER used for model fitting,
    only for calibration curve estimation.

    Uses one-vs-rest logistic calibration (Platt scaling) for multiclass.
    """
    # Step 1: Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    # Step 2: Train RBF SVM (no probability)
    svm = SVC(kernel='rbf', C=C, gamma='scale', probability=False,
              random_state=random_state)
    svm.fit(X_train_s, y_train)
    # Suppress sklearn warning about 'probability' parameter deprecation
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.svm")

    # Step 3: Get decision scores for calibration
    # decision_function returns shape (n_samples, n_classes) for multiclass
    train_scores = svm.decision_function(X_train_s)
    val_scores = svm.decision_function(X_val_s)

    # Handle potential shape issues (SVC decision_function may return 1D for binary)
    if train_scores.ndim == 1:
        train_scores = np.column_stack([-train_scores, train_scores])
    if val_scores.ndim == 1:
        val_scores = np.column_stack([-val_scores, val_scores])

    # Ensure we have n_classes columns
    if train_scores.shape[1] < n_components:
        padded = np.zeros((train_scores.shape[0], n_components))
        padded[:, :train_scores.shape[1]] = train_scores
        train_scores = padded
    if val_scores.shape[1] < n_components:
        padded = np.zeros((val_scores.shape[0], n_components))
        padded[:, :val_scores.shape[1]] = val_scores
        val_scores = padded

    # Step 4: Fit one-vs-rest logistic calibrators (Platt scaling)
    cal_clfs = []
    for c in range(n_components):
        # Binary problem: class c vs rest
        y_binary = (y_train == c).astype(int)

        clf = LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver='lbfgs',
            random_state=random_state
        )

        # Use the score for class c
        clf.fit(train_scores[:, c].reshape(-1, 1), y_binary)
        cal_clfs.append(clf)

    # Step 5: Apply calibration to validation fold
    proba_val = np.zeros((len(y_val), n_components))
    for c in range(n_components):
        proba_val[:, c] = cal_clfs[c].predict_proba(
            val_scores[:, c].reshape(-1, 1)
        )[:, 1]

    # Normalize to proper probabilities
    proba_val = np.clip(proba_val, 1e-12, 1.0 - 1e-12)
    proba_val = proba_val / proba_val.sum(axis=1, keepdims=True)

    calibration_info = {
        'svm_C': C,
        'n_train': len(y_train),
        'n_val': len(y_val),
        'calibration_method': 'one_vs_rest_platt',
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
    }

    return proba_val, calibration_info, scaler, svm, cal_clfs


def full_nested_rbf_cv(X_full, y_full, fold_assignments, n_components=3, C=1.0):
    """
    Perform nested CV: each fold serves once as validation for calibration.

    Parameters:
    -----------
    X_full : np.ndarray, shape (N, D)
        Full feature matrix
    y_full : np.ndarray, shape (N,)
        Full labels
    fold_assignments : np.ndarray, shape (N,)
        Integer fold assignment for each sample (0 to k-1)
    n_components : int
        Number of classes
    C : float
        RBF SVM regularization

    Returns:
    --------
    oof_probas : np.ndarray, shape (N, n_classes)
        Out-of-fold calibrated probabilities
    fold_results : dict
        Per-fold calibration statistics
    """
    N = len(y_full)
    n_folds = len(np.unique(fold_assignments))
    oof_probas = np.zeros((N, n_components))
    fold_results = {}

    for fold_idx in range(n_folds):
        # Validation mask for this fold
        val_mask = fold_assignments == fold_idx
        train_mask = ~val_mask

        X_train, y_train = X_full[train_mask], y_full[train_mask]
        X_val, y_val = X_full[val_mask], y_full[val_mask]

        # Train with nested calibration
        proba_val, info, _, _, _ = train_rbf_with_nested_calibration(
            X_train, y_train, X_val, y_val,
            n_components=n_components, C=C
        )

        oof_probas[val_mask] = proba_val
        fold_results[fold_idx] = {
            **info,
            'fold': fold_idx,
            'n_val_samples': len(y_val),
            'classes_in_val': np.unique(y_val).tolist()
        }

    return oof_probas, fold_results


def compute_ternary_cross_entropy(probs, y_true, eps=1e-12):
    """Compute multiclass cross-entropy for calibration evaluation."""
    probs = np.clip(probs, eps, 1.0 - eps)
    probs = probs / probs.sum(axis=1, keepdims=True)

    n_classes = probs.shape[1]
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(y_true)), y_true] = 1.0

    log_probs = np.log(probs)
    nll = -np.mean(np.sum(log_probs * one_hot, axis=1))
    return nll


def compute_brier_score(probs, y_true, eps=1e-12):
    """Compute multiclass Brier score."""
    probs = np.clip(probs, eps, 1.0 - eps)

    # Create one-hot encoding
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(y_true)), y_true] = 1.0

    brier = np.mean(np.sum((probs - one_hot) ** 2, axis=1))
    return brier


def ece_with_bins(probs, y_true, n_bins=10):
    """
    Expected Calibration Error with detailed binning.
    Returns (ECE, bin_data_dict)
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim == 1:
        probs = probs.reshape(-1, 1)

    # Get max probability and predicted class
    max_prob = np.max(probs, axis=1)
    pred_class = np.argmax(probs, axis=1)

    # Compute accuracy per bin
    correct = (pred_class == y_true).astype(float)

    # Bin boundaries
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_data = {}

    ece_val = 0.0
    total_samples = len(probs)

    for i in range(n_bins):
        bin_mask = (max_prob > bin_boundaries[i]) & (max_prob <= bin_boundaries[i + 1])
        bin_count = int(bin_mask.sum())

        if bin_count > 0:
            bin_accuracy = float(correct[bin_mask].mean())
            bin_confidence = float(max_prob[bin_mask].mean())
            ece_val += bin_count * abs(bin_accuracy - bin_confidence)

            bin_data[i] = {
                'bin_start': round(bin_boundaries[i], 4),
                'bin_end': round(bin_boundaries[i + 1], 4),
                'count': bin_count,
                'avg_confidence': round(bin_confidence, 4),
                'empirical_accuracy': round(bin_accuracy, 4),
                'calibration_gap': round(abs(bin_accuracy - bin_confidence), 4)
            }
        else:
            bin_data[i] = {
                'bin_start': round(bin_boundaries[i], 4),
                'bin_end': round(bin_boundaries[i + 1], 4),
                'count': 0,
                'avg_confidence': 0.0,
                'empirical_accuracy': 0.0,
                'calibration_gap': 0.0
            }

    return round(ece_val / total_samples, 8), bin_data


def generate_reliability_diagram(probs, y_true, ax=None, title='Reliability Diagram'):
    """Generate ECE reliability diagram with plotting."""
    from matplotlib import pyplot as plt

    ece_val, bin_data = ece_with_bins(probs, y_true, n_bins=10)

    bin_mids = []
    accuracies = []
    counts = []

    for i, data in bin_data.items():
        bin_mids.append((data['bin_start'] + data['bin_end']) / 2)
        accuracies.append(data['empirical_accuracy'])
        counts.append(data['count'])

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    ax.bar(bin_mids, accuracies, width=0.09, alpha=0.7,
           label='Empirical accuracy', color='#2b5c8f')
    ax.plot([0, 1], [0, 1], 'r--', label='Perfect calibration', linewidth=1.5)
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'{title}\nECE = {ece_val:.4f}')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    return ece_val, bin_data


if __name__ == '__main__':
    print("Phase 7 Nested Calibration Module")
    print("=" * 50)