# Unresolved Issues, Warnings, and Missing Evidence Audit

This document explicitly identifies all technical qualifications, historical discrepancies, warnings, and unverified requirements present in the repository. The final IEEE research paper must present these facts transparently.

---

## 1. Summary of Audit Classifications

From the independent numerical verification matrix (inal_verification_matrix.csv):
- **Checks Passed:** 163 (100% of executable functional, empirical, and mathematical checks)
- **Checks Failed:** 0
- **Documented Warnings:** 4
- **Unverified Requirements:** 3

---

## 2. Detailed Audit Warnings (Disclosed Technical Nuances)

### Warning 1: Calibration Quality vs Transition Recall Trade-off
- **Finding:** While the selected Phase 7 model (**E4**, $\\alpha = 0.75$, balanced logistic regression) substantially improves transition detection (.89\\%$ recall vs .43\\%$ for F3), its probability calibration degrades.
- **Evidence:** Brier score increases from .1814$ (E2) to .2372$ (E4). Expected Calibration Error (ECE) increases from .0570$ to .0682$.
- **Cause:** Balanced class weighting expands logistic probability outputs towards extreme values ($ and $) to overcome the severe $ class imbalance, distorting probability calibration while optimizing the harmonic mean objective $.
- **Recommendation for IEEE Paper:** Explicitly report this as a fundamental modeling trade-off between event sensitivity and probability sharpness, rather than claiming an unmitigated win.

### Warning 2: Preserved In-Sample Platt Calibration in Historical Nested CV
- **Finding:** In the separate, pre-existing Phase7_Nested_CV_Calibration/source/nested_calibration.py, the function 	rain_rbf_with_nested_calibration fits its per-class logistic calibrators using:
  `python
  clf.fit(train_scores[:, c].reshape(-1, 1), y_binary)
  `
  where 	rain_scores is computed from the training samples (svm.decision_function(X_train_s)).
- **Impact:** The calibrator fitting within this historical experiment is in-sample, not strictly out-of-sample as described in its text documentation.
- **Status:** Under the read-only rule, this historical file was preserved unchanged. It is NOT reused in the new Phase 7 precursor implementation.
- **Recommendation for IEEE Paper:** Do not cite the nested calibration numbers as pure out-of-fold calibrated benchmarks without noting this code implementation feature.

### Warning 3: Preserved Legacy Precursor Source Files
- **Finding:** An earlier, incomplete Phase 7 precursor attempt existed in the repository prior to this implementation (
un_phase7_precursor.py and source/precursor_features.py).
- **Characteristics:** The legacy code hardcoded global PM2.5 mean/std (.99, 81.13$), misidentified the target station, and lacked controlled E0–E7 ablations.
- **Resolution:** These legacy files were preserved byte-for-byte to honor the strict read-only mandate. The new reproducible implementation uses source/physical_precursors.py and phase7_run.py.
- **Recommendation for IEEE Paper:** Ensure the paper references only the corrected source/physical_precursors.py implementation.

### Warning 4: Single Primary Literature Access Limitation
- **Finding:** Reference [12] (H. R. Künsch 1989, *The Jackknife and the Bootstrap for General Stationary Observations*, Annals of Statistics) was verified via secondary citations and DOI metadata, but primary publisher full-text access was rate-limited/blocked during automated validation.
- **Resolution:** The mathematical equations for block bootstrap are standard and independently verified via numpy implementation.
- **Recommendation for IEEE Paper:** Manually review the full PDF of Künsch (1989) before final camera-ready submission.

---

## 3. Unverified Items (Evidence Gaps Acknowledged)

### Item 1: Original Professor-Issued P2–P7 Rubric
- **Status:** NOT VERIFIED (Authoritative original prompt/rubric not supplied in repository).
- **Finding:** The project repository contains working descriptions of syllabus requirements P2–P7, but the original professor syllabus document was not provided.
- **Resolution:** A comprehensive compliance audit was mapped to working descriptions in P2_P7_COMPLIANCE_AUDIT.md.

### Item 2: Original 3,168-Feature Raw SVM Baseline
- **Status:** NOT VERIFIED (Artifact not present in historical repository).
- **Finding:** An SVM trained on the uncompressed ,168$-dimensional raw tensor windows ( \\times 12 \\times 11$) was never saved in Phase 5 or Phase 6. All SVM models were trained on compressed PCA or HoSVD features.
- **Resolution:** Standalone Linear SVM on PCA-C (=576$) and PCA-B (=192$) are used as the verified upper-bound baselines. No fabricated 3,168-feature numbers are invented.

### Item 3: Row-Level Predictions from Historical Nested CV Experiment
- **Status:** NOT VERIFIED (Predictions not saved).
- **Finding:** Phase7_Nested_CV_Calibration saved summary CSVs and JSONs (phase7_comparison.csv, phase7_summary.json), but did not export row-level predictions for test samples.
- **Resolution:** Its aggregate numbers are documented in 
ested_calibration_historical_unverified.csv and classified as **NOT DIRECTLY COMPARABLE** due to Scope B evaluation conventions.

---

## 4. Population and Scope Discrepancy Reconciliation

| Experiment / Phase | Evaluated Split | Evaluated Scope | N_Samples | Actual Changes | Markov Accuracy | Standalone SVM Accuracy | Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Phase 6 Fusion** | Test | Scope A | 5,685 | 526 | 90.19% (unpredicted class macro) | 83.74% (Linear Best) | VERIFIED |
| **Phase 7 Precursors** | Test | Scope A | 5,685 | 526 | 90.75% (3-class macro) | 83.69% (Linear Best Scope A) | VERIFIED |
| **Historical Nested CV** | Test | Scope B (labeled Scope A) | 5,720 | 561 (includes 35 missing) | 90.42% (5172/5720) | 83.74% (full split) | INCONSISTENT |

- **Root Cause:** Historical Nested CV evaluated on all ,720$ Scope B samples while setting missing current states (=35$, labeled -1) as incorrect transitions, yielding an apparent accuracy of ,172 / 5,720 = 90.4196\\%$.
- **Policy:** The final IEEE paper must report results strictly on verified Scope A (=5,685$) with Scope B fallback reported in secondary columns.
