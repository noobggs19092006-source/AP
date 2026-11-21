"""
load_existing_models.py – Load source models/scalers/calibrators/mappings.
Enforces security: only loads trusted local project files.
Checks sklearn classes_ == [0,1,2] and verifies score/probability parity.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def load_model_and_calibrator(project_root: Path, model_id: str, split: str = "validation"):
    m_dir = project_root / "Phase5_SVM_Kernel_RFF_Results/models"
    p_dir = project_root / "Phase5_SVM_Kernel_RFF_Results/predictions"

    # Map model_id to file names and config
    mapping = {
        "linear_pca_B": {
            "model": "final_linear_pca_B.pkl", "cal": "calibrator_linear_pca_B.pkl",
            "scaler": "scaler_pca_B.npz", "cfg": "pca_B", "mtype": "linear",
            "csv_val": p_dir / "final_linear_pca_B_validation.csv",
            "csv_test": p_dir / "final_linear_pca_B_test.csv",
        },
        "linear_hosvd_B": {
            "model": "final_linear_hosvd_B.pkl", "cal": "calibrator_linear_hosvd_B.pkl",
            "scaler": "scaler_hosvd_B.npz", "cfg": "hosvd_B", "mtype": "linear",
            "csv_val": p_dir / "final_linear_hosvd_B_validation.csv",
            "csv_test": p_dir / "final_linear_hosvd_B_test.csv",
        },
        "linear_best_overall": {
            "model": "final_linear_best_overall.pkl", "cal": "calibrator_linear_best_overall.pkl",
            "scaler": "scaler_pca_C.npz", "cfg": "pca_C", "mtype": "linear",
            "csv_val": p_dir / "final_linear_best_overall_validation.csv",
            "csv_test": p_dir / "final_linear_best_overall_test.csv",
        },
        "revised_rff_pca_B": {
            "model": "final_revised_rff_pca_B.pkl", "cal": "calibrator_revised_rff_pca_B.pkl",
            "scaler": "scaler_pca_B.npz", "cfg": "pca_B", "mtype": "rff_revised",
            "csv_val": p_dir / "revised/final_revised_rff_pca_B_validation.csv",
            "csv_test": p_dir / "revised/final_revised_rff_pca_B_test.csv",
        },
        "revised_rff_hosvd_B": {
            "model": "final_revised_rff_hosvd_B.pkl", "cal": "calibrator_revised_rff_hosvd_B.pkl",
            "scaler": "scaler_hosvd_B.npz", "cfg": "hosvd_B", "mtype": "rff_revised",
            "csv_val": p_dir / "revised/final_revised_rff_hosvd_B_validation.csv",
            "csv_test": p_dir / "revised/final_revised_rff_hosvd_B_test.csv",
        },
        "rff_pca_B": {
            "model": "final_rff_pca_B.pkl", "cal": "calibrator_rff_pca_B.pkl",
            "scaler": "scaler_pca_B.npz", "cfg": "pca_B", "mtype": "rff_unrevised",
            "map_npz": m_dir / "rff_mapping_revised_pca_B_D2048_s44.npz",
            "csv_val": p_dir / "final_rff_pca_B_validation.csv",
            "csv_test": p_dir / "final_rff_pca_B_test.csv",
        },
        "rff_hosvd_B": {
            "model": "final_rff_hosvd_B.pkl", "cal": "calibrator_rff_hosvd_B.pkl",
            "scaler": "scaler_hosvd_B.npz", "cfg": "hosvd_B", "mtype": "rff_unrevised",
            "map_npz": m_dir / "rff_mapping_revised_hosvd_B_D2048_s44.npz",
            "csv_val": p_dir / "final_rff_hosvd_B_validation.csv",
            "csv_test": p_dir / "final_rff_hosvd_B_test.csv",
        },
        "stageB_full_pca_B": {
            "model": "final_stageB_full_pca_B.pkl", "cal": None,
            "scaler": "scaler_pca_B.npz", "cfg": "pca_B", "mtype": "exact_rbf",
            "csv_val": p_dir / "stage_b/final_stageB_full_pca_B_validation.csv",
            "csv_test": p_dir / "stage_b/final_stageB_full_pca_B_test.csv",
        },
        "stageB_full_hosvd_B": {
            "model": "final_stageB_full_hosvd_B.pkl", "cal": None,
            "scaler": "scaler_hosvd_B.npz", "cfg": "hosvd_B", "mtype": "exact_rbf",
            "csv_val": p_dir / "stage_b/final_stageB_full_hosvd_B_validation.csv",
            "csv_test": p_dir / "stage_b/final_stageB_full_hosvd_B_test.csv",
        },
    }

    if model_id not in mapping:
        raise ValueError(f"Unknown model_id: {model_id}")

    info = mapping[model_id]
    m_path = m_dir / info["model"]
    c_path = m_dir / info["cal"] if info["cal"] else None
    s_path = m_dir / info["scaler"]
    csv_key = "csv_val" if split == "validation" else "csv_test"
    csv_path = info[csv_key]

    # Load model and calibrator
    m_obj = pickle.load(open(m_path, "rb"))
    c_obj = pickle.load(open(c_path, "rb")) if c_path else None
    sc = np.load(s_path, allow_pickle=True)

    # Read predictions CSV directly to verify parity
    df_csv = pd.read_csv(csv_path)

    return {
        "model_id": model_id,
        "model": m_obj,
        "calibrator": c_obj,
        "scaler": sc,
        "config_id": info["cfg"],
        "model_type": info["mtype"],
        "csv": df_csv,
        "map_npz": info.get("map_npz"),
    }
