"""
cost_sensitive_training.py – Cost-sensitive SVM/RFF training for transition events.
Implements sample-weighted loss for rare transition detection.
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pathlib import Path
import pickle


def compute_sample_weights(y_true, current_state, transition_weights=None):
    """
    Compute sample weights for cost-sensitive learning.

    Parameters:
    -----------
    y_true : array, shape (N,)
        True next-state labels
    current_state : array, shape (N,)
        Observed current states
    transition_weights : dict, optional
        Custom weights per transition type:
        {
            'transition': weight for change events,
            'stable': weight for stable events,
            'high_entry': weight for high-entry detection
        }

    Returns:
    --------
    weights : np.ndarray, shape (N,)
        Sample weights for training
    """
    if transition_weights is None:
        transition_weights = {
            'transition': 5.0,  # Weight for detected changes
            'stable': 1.0,       # Weight for stable hours
            'high_entry': 10.0   # Weight for high-entry events
        }

    is_transition = (y_true != current_state)
    is_high_entry = ((current_state != 2) & (y_true == 2))

    weights = np.ones(len(y_true))

    # Apply transition weight
    weights[is_transition] = transition_weights['transition']

    # Apply high-entry weight
    weights[is_high_entry] = transition_weights['high_entry']

    # Stable hours get baseline weight
    weights[~is_transition] = transition_weights['stable']

    return weights


def cost_sensitive_svm(X, y, sample_weights, C=1.0, kernel='rbf', class_weight=None,
                       gamma='scale', probability=False):
    """
    Train cost-sensitive SVM using sample weights.

    Parameters:
    -----------
    X : np.ndarray, shape (N, D)
        Feature matrix
    y : np.ndarray, shape (N,)
        Labels
    sample_weights : np.ndarray
        Per-sample weights
    C : float
        Regularization parameter
    kernel : str
        SVM kernel
    class_weight : dict or None
        Built-in class weight (overrides if provided)
    gamma : str or float
        RBF kernel gamma
    probability : bool
        Return probability estimates

    Returns:
    --------
    model : trained SVM
    scaler : fitted StandardScaler
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        probability=probability,
        class_weight=class_weight,
        random_state=42
    )

    model.fit(X_scaled, y, sample_weight=sample_weights)

    return model, scaler


def cost_sensitive_logistic(X, y, sample_weights, C=1.0):
    """
    Train cost-sensitive logistic regression.

    Parameters:
    -----------
    X : np.ndarray, shape (N, D)
        Feature matrix
    y : np.ndarray, shape (N,)
        Labels
    sample_weights : np.ndarray
        Per-sample weights
    C : float
        Inverse regularization

    Returns:
    --------
    model : LogisticRegression with sample weights applied
    scaler : StandardScaler
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # sklearn LogisticRegression supports sample_weight in fit()
    model = LogisticRegression(
        C=C,
        max_iter=1000,
        multi_class='multinomial',
        solver='lbfgs',
        random_state=42
    )

    model.fit(X_scaled, y, sample_weight=sample_weights)

    return model, scaler


def cost_sensitive_sgd(X, y, sample_weights, alpha=0.01, learning_rate='optimal'):
    """
    Train cost-sensitive linear classifier using SGD.
    More scalable for large datasets.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = SGDClassifier(
        alpha=alpha,
        loss='log',
        learning_rate=learning_rate,
        random_state=42
    )

    model.fit(X_scaled, y, sample_weight=sample_weights)

    return model, scaler


def evaluate_weight_sensitivity(X_train, y_train, current_state_train,
                                  X_val, y_val, current_state_val,
                                  weight_factors=[1, 2, 5, 10, 20]):
    """
    Evaluate model performance across different transition weight factors.

    Returns dict with performance metrics for each weight factor.
    """
    from evaluation import compute_metrics, ensure_prob_simplex

    results = {}

    base_weights = compute_sample_weights(y_train, current_state_train)
    val_weights = compute_sample_weights(y_val, current_state_val)

    for w in weight_factors:
        # Scale transition weights
        adjusted_weights = base_weights.copy()
        transition_mask = (y_train != current_state_train)
        adjusted_weights[transition_mask] *= w

        model, scaler = cost_sensitive_svm(
            X_train, y_train, adjusted_weights,
            C=1.0, kernel='rbf', probability=True
        )

        X_val_scaled = scaler.transform(X_val)
        probs = model.predict_proba(X_val_scaled)
        probs = ensure_prob_simplex(probs)
        preds = model.predict(X_val_scaled)

        metrics = compute_metrics(y_val, preds, current_state_val, probs=probs)
        metrics['weight_factor'] = w
        metrics['validation_weight_sum'] = float(val_weights.sum())

        results[w] = metrics

    return results


def find_optimal_weights(X_train, y_train, current_state_train,
                          X_val, y_val, current_state_val,
                          weight_grid=None):
    """
    Grid search over weight parameters for cost-sensitive training.
    """
    if weight_grid is None:
        weight_grid = {
            'transition': [2, 5, 10, 15],
            'stable': [1.0],
            'high_entry': [5, 10, 15]
        }

    best_J = -np.inf
    best_params = None
    all_results = []

    from evaluation import compute_metrics, ensure_prob_simplex

    transition_vals = weight_grid['transition']
    stable_vals = weight_grid['stable']
    high_entry_vals = weight_grid['high_entry']

    for t_w in transition_vals:
        for s_w in stable_vals:
            for h_w in high_entry_vals:
                tw = compute_sample_weights(y_train, current_state_train, {
                    'transition': t_w, 'stable': s_w, 'high_entry': h_w
                })

                model, scaler = cost_sensitive_svm(
                    X_train, y_train, tw, C=1.0, kernel='rbf', probability=True
                )

                X_val_scaled = scaler.transform(X_val)
                probs = model.predict_proba(X_val_scaled)
                probs = ensure_prob_simplex(probs)
                preds = model.predict(X_val_scaled)

                metrics = compute_metrics(y_val, preds, current_state_val, probs=probs)
                J = 0.5 * metrics['macro_f1'] + 0.5 * metrics['change_F1']

                result = {
                    'transition_weight': t_w,
                    'stable_weight': s_w,
                    'high_entry_weight': h_w,
                    'accuracy': metrics['accuracy'],
                    'macro_f1': metrics['macro_f1'],
                    'change_F1': metrics['change_F1'],
                    'A_change': metrics['A_change'],
                    'high_entry_F1': metrics['high_F1'],
                    'J': round(J, 8)
                }
                all_results.append(result)

                if J > best_J:
                    best_J = J
                    best_params = {
                        'transition': t_w,
                        'stable': s_w,
                        'high_entry': h_w
                    }

    return best_params, best_J, all_results


if __name__ == '__main__':
    print("Phase 7 Cost-Sensitive Training Module")
    print("=" * 50)