# Phase 6 Limitations and Caveats

## 1. Test Set Exposure (Exploratory Status)
**Phase 5 test results have been previously examined.** Phase 6 is a post-hoc exploratory analysis, not an independent confirmatory evaluation. All fusion design choices were informed by knowledge of Phase 5 standalone SVM performance (e.g., change recall ~50%, accuracy ~83–84%).

## 2. Calibration Isolation Not Fully Nested
Phase 5 calibration used a **single chronological 80/20 holdout** within the full training partition (not forward-chaining / rolling-origin). The same calibration labels are reused across all validation comparisons. This means:
- Calibration quality metrics (Brier, LogLoss, ECE) on validation/test are optimistically biased.
- Phase 6 fusion experiments inherit this bias without additional correction.

## 3. Phase 4 Bases Inherited from Full Training
PCA loadings and HoSVD factor matrices were fitted on the **entire training period** (24,108 samples). No nested re-fitting within cross-validation folds was performed. The pipeline is therefore **not fully nested** in the strict statistical sense.

## 4. Exact-RBF Probability Fusion Unsupported
Exact RBF kernel SVMs (`stageB_full_pca_B`, `stageB_full_hosvd_B`) lack calibrated class probabilities. Implementing train-only out-of-sample calibration would require:
- Refitting RBF SVCs on OOF training folds (computationally expensive: ~17s/fit on full train, ~9s/fit on 80%).
- Generating OOF decision margins for Platt/multinomial calibration.
This was **not done** due to computational cost. Exact RBF models are compared on **hard labels only** in Phase 6.

## 5. scikit-learn Version Compatibility
Phase 5 models were pickled with `scikit-learn 1.9.1`. Reloading may emit `InconsistentVersionWarning`. Prediction parity was verified for all linear/RFF models; exact RBF parity on test hard predictions was verified.

## 6. Temporal Dependence & Bootstrap Assumptions
The paired time-block bootstrap uses **contiguous 24-hour blocks** as resampling units. This assumes:
- Within-block dependence is strong; between-block dependence is negligible.
- 24 hours is sufficient to capture diurnal autocorrelation.
Block length sensitivity was **not explored**; alternative block lengths (168 hours / 1 week) were not tested.

## 7. Uneven Class / Transition Frequencies
Test Scope A: 526 true transitions out of 5,685 hours (9.25%). Specific transitions are highly imbalanced:
- Low → Moderate: most common
- Low → High: rare (154 High-entry events)
- Moderate → High: rare
Macro-F1 and Change-F1 give equal weight to rare transitions; results may be unstable under small count perturbations.

## 8. Missing Current States (Scope B Fallback)
35 test rows lack observed current state (`current_state = -1`). Scope B evaluation uses feature-only SVM fallback without Markov probabilities. This fallback is **not validated** and may produce unreliable predictions for those rows.

## 9. First-Order Markov Assumption
The transition matrix $P$ assumes $P(S_{t+1} \mid S_t, S_{t-1}, \dots) = P(S_{t+1} \mid S_t)$. Phase 3 adequacy tests showed marginal second-order effects but these were **not incorporated** into Phase 6 fusion.

## 10. Single Target Station / Single Year Range
All results are for **Aotizhongxin station only**, March 2013 – February 2017. Generalization to other stations, cities, or seasons is **not evaluated**.

## 11. Multiple Validation Comparisons (Selection Sensitivity)
Grid searches evaluate 77 (F1) + 375 (F2) + 125 (F3) = 577 configurations. Selection sensitivity to grid granularity and tie-breaking rules was **not quantified** (e.g., ±0.05 lambda step vs ±0.1).

## 12. Exact-RBF Omitted from Probability Fusion Experiments
All probability-based fusion experiments (F1, F2, F3) used only calibrated Linear/RFF sources. The two best Phase 5 models (Exact RBF on PCA_B and HoSVD_B) **could not participate** in probabilistic fusion due to lack of calibration.

## 13. No Cost-Sensitive Retraining in Phase 6
Phase 5 explored sample-weight cost-sensitive training (λ = 2, 5, 10 on Scope A transitions). Phase 6 does **not** re-train SVM/RFF models with change-weighted losses; fusion operates on frozen calibrated probabilities only.

## 14. Uncompleted Optional Experiments
The following items specified as "optional" in the handover were **not executed**:
- Binary change logistic regression in F3 (only blended $q_t$ used).
- Explicit comparison of PCA vs HoSVD within same fusion family (grids mixed sources).
- ECE bin-level reliability diagrams (only scalar ECE reported).
- Phase 2 raw tensor loading (not required; all features/scalers/calibrators pre-loaded).

Each omission is a **deliberate resource allocation choice**, not a technical blocker. The Phase 6 package contains all code scaffolds to enable these extensions in Phase 7.