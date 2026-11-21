# Phase 6 — Transition-Aware Markov–SVM Fusion for Next-Hour Air Pollution Prediction

## Overview
This directory implements Phase 6 of the Beijing multi-site air-quality prediction
pipeline. It combines Phase 3 Markov state-transition probabilities with
Phase 5 calibrated SVM/RFF probability vectors using three fusion strategies:

* **F1** – Fixed-weight convex mixture `p_F = λ·p_S + (1−λ)·p_M`
* **F2** – Adaptive transition-risk gate: `λ_t = λ_high` if `r_S,t ≥ τ` else `λ_low`
* **F3** – Explicit change likelihood `q_t` + conditional destination `d_F,t`

## Required Inputs (read-only, not modified)
* `Air_Pollution_Phase3_Results/phase3_output/markov_transition_matrix.csv` – train-fitted 3×3 Markov P
* `phase4_output/features_{train,validation,test}_{pca_B,hosvd_B,pca_C}.npz` – Phase 4 compressed features
* `Phase5_SVM_Kernel_RFF_Results/` – saved models, calibrators, scalers, predictions

## How to Run
```bash
pip install -r requirements.txt
python Phase6_Markov_SVM_Fusion_Results/phase6_run.py
```
All outputs are written under `Phase6_Markov_SVM_Fusion_Results/`.

## Outputs
* `config/` – experiment spec, fusion grids, locked configuration
* `source/` – integrity gate, model loader, Markov/SVM probabilities, fusion modules, evaluation, plotting
* `models/` – any newly fitted calibration/binary-change models (none required here)
* `results/` – integrity report, probability audit, grid tables, final evaluation, transition detection, classwise metrics, confusion matrices, probability quality, bootstrap CIs, computational cost
* `predictions/validation/`, `predictions/test/` – per-row prediction CSVs
* `figures/` – nine scientific figures
* `reports/` – mathematical chapter, research report, limitations report
* `manifests/` – provenance, model selection, artifact inventory
* `tests/` – pytest suite
* `logs/` – execution log, pytest results

## Key Conventions
* Class order is always `[0, 1, 2]` = Low / Moderate / High
* Scope A = rows with observed current state (`current_state ≥ 0`); Scope B = all rows
* All probabilities are clipped to `EPS_LOGLOSS = 1e-12` and renormalized before log loss
* Model selection is **validation-only**; test labels never influence any fitted parameter
* Exact-RBF models (`stageB_full_*`) have no calibrated probabilities and are compared on hard labels only
* Phase 1–5 artifacts are treated as read-only; all new material lives under `Phase6_Markov_SVM_Fusion_Results/`