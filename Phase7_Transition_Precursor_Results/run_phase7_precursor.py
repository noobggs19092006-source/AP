#!/usr/bin/env python3
"""
run_phase7_precursor.py – Main execution script for Phase 7 Transition-Precursor-Aware Fusion.

This script:
1. Loads Phase 2 and Phase 6 data
2. Extracts precursor features
3. Trains transition classifier
4. Runs E0-E7 experiments
5. Generates all results, figures, and reports
6. Creates IEEE paper content and independent verification package
"""
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import time
import sys
import warnings
warnings.filterwarnings('ignore')

# Setup paths
PROJECT_ROOT = Path('C:/Users/aryan/Desktop/MATLAB/Project/AP')
PHASE7_DIR = PROJECT_ROOT / 'Phase7_Transition_Precursor_Results'

sys.path.insert(0, str(PHASE7_DIR / 'source'))
sys.path.insert(0, str(PROJECT_ROOT / 'Phase6_Markov_SVM_Fusion_Results' / 'source'))

from precursor_features import extract_all_precursor_features, compute_precursor_change_probability
from transition_classifier import TransitionClassifier, compute_change_labels
from precursor_fusion import run_e0_e7_experiments, compute_j_score
from evaluation import compute_metrics, ensure_prob_simplex, ece

# PM2.5 denormalization parameters
PM25_MEAN = 84.99
PM25_STD = 81.13


def load_phase2_data():
    """Load Phase 2 window data."""
    train = np.load(PROJECT_ROOT / 'Phase 2/phase2_output/train_windows.npz')
    val = np.load(PROJECT_ROOT / 'Phase 2/phase2_output/validation_windows.npz')
    test = np.load(PROJECT_ROOT / 'Phase 2/phase2_output/test_windows.npz')

    return {'train': train, 'val': val, 'test': test}


def load_phase6_predictions():
    """Load Phase 6 test predictions."""
    pred_path = PROJECT_ROOT / 'Phase6_Markov_SVM_Fusion_Results' / 'predictions' / 'test' / 'test_predictions_merged.csv'
    df = pd.read_csv(pred_path)
    return df


def load_markov_model():
    """Load Markov transition model."""
    from markov_probabilities import MarkovProbabilities
    markov = MarkovProbabilities(PROJECT_ROOT / 'Air_Pollution_Phase3_Results' / 'phase3_output' / 'markov_transition_matrix.csv')
    return markov


def main():
    """Main execution function."""
    print("=" * 70)
    print("PHASE 7: Transition-Precursor-Aware Markov–SVM Fusion")
    print("=" * 70)
    start_time = time.time()

    # Create output directories
    output_dirs = {
        'features': PHASE7_DIR / 'features',
        'models': PHASE7_DIR / 'models',
        'results': PHASE7_DIR / 'results',
        'predictions': PHASE7_DIR / 'predictions',
        'figures': PHASE7_DIR / 'figures',
        'diagrams': PHASE7_DIR / 'diagrams',
        'reports': PHASE7_DIR / 'reports',
        'literature': PHASE7_DIR / 'literature',
        'logs': PHASE7_DIR / 'logs',
        'tests': PHASE7_DIR / 'tests'
    }

    for dir_path in output_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Load Phase 2 data
    print("\n[1] Loading Phase 2 data...")
    phase2_data = load_phase2_data()
    print(f"    Train: {phase2_data['train']['X'].shape}")
    print(f"    Val:   {phase2_data['val']['X'].shape}")
    print(f"    Test:  {phase2_data['test']['X'].shape}")

    # Step 2: Load Phase 6 predictions
    print("\n[2] Loading Phase 6 predictions...")
    phase6_pred = load_phase6_predictions()
    print(f"    Loaded {len(phase6_pred)} predictions")

    # Step 3: Load Markov model
    print("\n[3] Loading Markov transition model...")
    markov_model = load_markov_model()

    # Step 4: Extract precursor features
    print("\n[4] Extracting precursor features...")
    train_features = extract_all_precursor_features(phase2_data['train']['X'], mean=PM25_MEAN, std=PM25_STD)
    val_features = extract_all_precursor_features(phase2_data['val']['X'], mean=PM25_MEAN, std=PM25_STD)
    test_features = extract_all_precursor_features(phase2_data['test']['X'], mean=PM25_MEAN, std=PM25_STD)
    print(f"    Train features: {train_features.shape}")
    print(f"    Val features:   {val_features.shape}")
    print(f"    Test features:  {test_features.shape}")

    # Save features
    np.savez_compressed(output_dirs['features'] / 'train_features.npz', features=train_features)
    np.savez_compressed(output_dirs['features'] / 'val_features.npz', features=val_features)
    np.savez_compressed(output_dirs['features'] / 'test_features.npz', features=test_features)

    # Step 5: Prepare Phase 6 fusion probabilities for evaluation
    print("\n[5] Preparing Phase 6 fusion probabilities...")

    # Load Markov probabilities
    p_markov_test = markov_model.get_probs(phase2_data['test']['current_state'])

    # Load SVM probabilities from Phase 6
    p_svm_test_0 = phase6_pred[['p_svm_0', 'p_svm_1', 'p_svm_2']].values
    p_f1_test = phase6_pred[['p_f1_0', 'p_f1_1', 'p_f1_2']].values
    p_f2_test = phase6_pred[['p_f2_0', 'p_f2_1', 'p_f2_2']].values
    p_f3_test = phase6_pred[['p_f3_0', 'p_f3_1', 'p_f3_2']].values

    # Step 6: Compute precursor change probabilities
    print("\n[6] Computing precursor change probabilities...")
    y_change_train = compute_change_labels(phase2_data['train']['y'], phase2_data['train']['current_state'])
    y_change_val = compute_change_labels(phase2_data['val']['y'], phase2_data['val']['current_state'])
    y_change_test = compute_change_labels(phase2_data['test']['y'], phase2_data['test']['current_state'])

    # Compute precursor change probabilities using heuristic method
    # (threshold proximity + trends)
    q_precursor_train = compute_precursor_change_probability(train_features)
    q_precursor_val = compute_precursor_change_probability(val_features)
    q_precursor_test = compute_precursor_change_probability(test_features,
                                                               train_y_change=y_change_train,
                                                               train_features=train_features)

    # Also save the transition classifier for reference
    clf = TransitionClassifier(C=1.0, random_state=42)
    clf.fit(train_features, y_change_train)
    with open(output_dirs['models'] / 'transition_classifier.pkl', 'wb') as f:
        pickle.dump({'clf': clf, 'feature_names': clf.feature_names}, f)
    print(f"    Classifier and precursor probabilities computed")
    print(f"    q_precursor (heuristic): mean={q_precursor_test.mean():.4f}, std={q_precursor_test.std():.4f}")

    # Debug: compare with SVM change probability
    r_svm = 1.0 - np.take_along_axis(p_svm_test_0, phase2_data['test']['current_state'][:, None].astype(int), axis=1).squeeze(-1)
    print(f"    r_svm: mean={r_svm.mean():.4f}, std={r_svm.std():.4f}")

    # Step 7: Run E0-E7 experiments
    print("\n[7] Running E0-E7 experiments...")

    # Run experiments with precursor features for destination bias
    results = run_e0_e7_experiments(
        p_svm=p_svm_test_0,
        p_markov=p_markov_test,
        y_true=phase2_data['test']['y'],
        current_state=phase2_data['test']['current_state'],
        p_f3=p_f3_test,
        q_precursor=q_precursor_test,
        output_dir=output_dirs['results'],
        precursor_features=test_features
    )

    # Step 8: Compute metrics and J-scores
    print("\n[8] Computing metrics...")
    j_scores = {}
    for r in results:
        j = compute_j_score(r)
        j_scores[r['model']] = j
    print(f"    Best J-score: {max(j_scores.values()):.6f} ({max(j_scores, key=j_scores.get)})")

    # Step 9: Save predictions
    print("\n[9] Saving predictions...")
    pred_df = pd.DataFrame({
        'y_true': phase2_data['test']['y'],
        'current_state': phase2_data['test']['current_state'],
        'target_timestamp': phase2_data['test']['target_timestamp'],
        'q_precursor': q_precursor_test,
        'p_svm_0': p_svm_test_0[:, 0],
        'p_svm_1': p_svm_test_0[:, 1],
        'p_svm_2': p_svm_test_0[:, 2],
        'p_f1_0': p_f1_test[:, 0],
        'p_f1_1': p_f1_test[:, 1],
        'p_f1_2': p_f1_test[:, 2],
        'p_f2_0': p_f2_test[:, 0],
        'p_f2_1': p_f2_test[:, 1],
        'p_f2_2': p_f2_test[:, 2],
        'p_f3_0': p_f3_test[:, 0],
        'p_f3_1': p_f3_test[:, 1],
        'p_f3_2': p_f3_test[:, 2],
    })
    pred_df.to_csv(output_dirs['predictions'] / 'test_predictions.csv', index=False)

    # Step 10: Generate comparison table
    print("\n[10] Generating comparison table...")
    rows = []
    for r in results:
        rows.append({
            'model_id': r['model'],
            'accuracy': r['accuracy'],
            'macro_f1': r['macro_f1'],
            'change_F1': r['change_F1'],
            'A_change': r['A_change'],
            'A_stable': r['A_stable'],
            'J_score': compute_j_score(r),
            'ECE': r.get('ECE', 'N/A')
        })

    df_comparison = pd.DataFrame(rows)
    df_comparison.to_csv(output_dirs['results'] / 'phase7_comparison.csv', index=False)

    # Step 11: Save summary
    elapsed = time.time() - start_time
    summary = {
        'phase': 'Phase 7 - Transition-Precursor-Aware Fusion',
        'timestamp': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'elapsed_seconds': round(elapsed, 2),
        'best_model': max(j_scores, key=j_scores.get),
        'best_j_score': round(max(j_scores.values()), 6),
        'experiments': list(j_scores.keys()),
        'j_scores': {k: round(v, 6) for k, v in j_scores.items()},
        'thresholds': {'low': 35.0, 'high': 75.0},
        'precursor_features': ['d_35', 'd_75', 'd_min', 'v_3', 'v_6', 'cross_mean_trend', 'cross_max_trend', 'threshold_crossing_risk', 'time_pattern'],
        'fusion_params': {'alpha': 0.3, 'rho': 0.5, 'eta': 0.5}
    }

    with open(output_dirs['results'] / 'phase7_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"PHASE 7 COMPLETE in {elapsed:.2f}s")
    print(f"{'=' * 70}")
    print(f"\nResults saved to: {output_dirs['results']}")
    print(f"Best J-score: {summary['best_j_score']:.6f} ({summary['best_model']})")

    return summary


if __name__ == "__main__":
    main()