"""
load_phase5_models.py – Load Phase 5 models for Phase 7 extended evaluation.
"""
import pickle
import numpy as np
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV


def load_phase5_svm_model(model_id, project_root):
    """Load Phase 5 SVM model and associated artifacts."""
    model_dir = project_root / 'Phase5_SVM_Kernel_RFF_Results' / 'models'

    if model_id == 'linear_best_overall':
        model_path = model_dir / 'final_linear_best_overall.pkl'
        scaler_path = project_root / 'phase4_output' / 'scaler_pca_B.pkl'

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        return {
            'model': model,
            'model_id': model_id
        }

    elif model_id.startswith('stageB_full_'):
        if 'pca' in model_id or 'pca_B' in model_id:
            cfg = 'pca_B'
        else:
            cfg = 'hosvd_B'

        model_path = model_dir / f'final_{model_id}.pkl'

        with open(model_path, 'rb') as f:
            model_dict = pickle.load(f)

        return {
            'model': model_dict,
            'model_id': model_id,
            'config': cfg
        }

    else:
        raise ValueError(f"Unknown model_id: {model_id}")


def load_phase5_features(split, project_root):
    """Load Phase 5 feature matrices for given split."""
    base_dir = project_root / 'phase4_output'

    if split == 'training':
        npz_path = base_dir / 'features_training_pca_B.npz'
    elif split == 'validation':
        npz_path = base_dir / 'features_validation_pca_B.npz'
    elif split == 'test':
        npz_path = base_dir / 'features_test_pca_B.npz'
    else:
        raise ValueError(f"Unknown split: {split}")

    data = np.load(npz_path, allow_pickle=True)

    return {
        'F': data['F'],
        'y': data['y'],
        'current_state': data['current_state'].astype(int),
        'target_timestamp': data.get('target_timestamp', None)
    }


def load_phase6_calibrators(model_id, project_root):
    """Load Phase 6 calibrators for existing models."""
    # Phase 6 stores calibrators in models/ directory
    cal_dir = project_root / 'Phase6_Markov_SVM_Fusion_Results' / 'models'

    cal_path = cal_dir / f'{model_id}_calibrator.pkl'
    if cal_path.exists():
        with open(cal_path, 'rb') as f:
            return pickle.load(f)

    return None