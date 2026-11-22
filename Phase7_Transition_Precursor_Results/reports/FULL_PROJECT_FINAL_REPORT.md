# Full project final report

The project compares training-only tensor compression, state-only Markov predictions, SVM variants and probability fusion. Earlier phases are retained unchanged. The new experiment asks whether interpretable physical precursor signals improve transition detection.

The locked choice is **E4**, classifier `temporal_C1.0_balanced`, alpha=0.75, with probability argmax decisions. It was selected using validation J=0.620206. The C grid was 0.1, 1 and 10; each was evaluated with no weighting and balanced weighting. E3/E4/E5 use 6/8/11 features respectively. E6 keeps E5's trained classifier and alpha and selects theta on validation. E7 fixes alpha=1.

## Matched comparison

All finite rows below are test Scope A, N=5,685; nested calibration is unverified and omitted from this numeric table.

| model | accuracy | macro_f1 | change_recall | change_f1 | exact_destination | stable_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| E0 | 0.90748 | 0.88951 | 0.00000 | 0.00000 | 0.00000 | 1.00000 |
| E1 | 0.83694 | 0.80233 | 0.46388 | 0.35184 | 0.41065 | 0.88040 |
| E2 | 0.88461 | 0.86615 | 0.26426 | 0.31699 | 0.15589 | 0.95891 |
| E3 | 0.83465 | 0.80566 | 0.55703 | 0.40582 | 0.40114 | 0.87885 |
| E4 | 0.83518 | 0.80742 | 0.55894 | 0.40777 | 0.40114 | 0.87943 |
| E5 | 0.84239 | 0.81727 | 0.57795 | 0.42757 | 0.42205 | 0.88525 |
| E6 | 0.84398 | 0.80235 | 0.52852 | 0.41216 | 0.34981 | 0.89436 |
| E7 | 0.78839 | 0.75988 | 0.71673 | 0.40299 | 0.55323 | 0.81237 |
| final_linear_best_overall | 0.83360 | 0.79037 | 0.50570 | 0.36893 | 0.43726 | 0.87401 |
| final_linear_hosvd_B | 0.80580 | 0.73230 | 0.54373 | 0.35550 | 0.41635 | 0.84551 |
| final_linear_pca_B | 0.82656 | 0.76172 | 0.50951 | 0.36314 | 0.42205 | 0.86780 |
| final_rff_hosvd_B | 0.79455 | 0.73953 | 0.52852 | 0.33514 | 0.40494 | 0.83427 |
| final_rff_pca_B | 0.80651 | 0.75258 | 0.51901 | 0.34405 | 0.40684 | 0.84726 |
| final_revised_rff_hosvd_B | 0.79437 | 0.73934 | 0.52852 | 0.33494 | 0.40494 | 0.83408 |
| final_revised_rff_pca_B | 0.79930 | 0.74020 | 0.52852 | 0.33861 | 0.42395 | 0.83757 |
| final_stageB_full_hosvd_B | 0.83694 | 0.81095 | 0.50380 | 0.37324 | 0.43346 | 0.87808 |
| final_stageB_full_pca_B | 0.84169 | 0.80938 | 0.48859 | 0.37219 | 0.42586 | 0.88409 |
| final_stageB_hosvd_B | 0.80264 | 0.75889 | 0.52662 | 0.34219 | 0.41825 | 0.84183 |
| final_stageB_pca_B | 0.79120 | 0.73702 | 0.54373 | 0.33687 | 0.42776 | 0.82826 |
| Phase6_f1 | 0.86051 | 0.83034 | 0.43916 | 0.37807 | 0.37643 | 0.90987 |
| Phase6_f2 | 0.86772 | 0.83905 | 0.33840 | 0.32781 | 0.29658 | 0.92595 |
| Phase6_f3 | 0.88461 | 0.86615 | 0.26426 | 0.31699 | 0.15589 | 0.95891 |

## Findings

E4 detects 294 of 526 changes (55.89%) versus 139 (26.43%) for F3. False alarms increase from 212 to 622. Accuracy changes from 88.46% to 83.52%, and macro-F1 from 0.8661 to 0.8074. Correct destination accuracy increases from 15.59% to 40.11%, while stable-state accuracy declines from 95.89% to 87.94%. This is a transition-sensitivity trade-off, not an overall accuracy improvement.

## Experiment details

| model | accuracy | macro_f1 | change_precision | change_recall | change_f1 | exact_destination | stable_accuracy | high_f1 | J |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | 0.90748 | 0.88951 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 1.00000 | 0.00000 | 0.44475 |
| E1 | 0.83694 | 0.80233 | 0.28339 | 0.46388 | 0.35184 | 0.41065 | 0.88040 | 0.30493 | 0.57709 |
| E2 | 0.88461 | 0.86615 | 0.39601 | 0.26426 | 0.31699 | 0.15589 | 0.95891 | 0.03436 | 0.59157 |
| E3 | 0.83465 | 0.80566 | 0.31917 | 0.55703 | 0.40582 | 0.40114 | 0.87885 | 0.30498 | 0.60574 |
| E4 | 0.83518 | 0.80742 | 0.32096 | 0.55894 | 0.40777 | 0.40114 | 0.87943 | 0.30263 | 0.60759 |
| E5 | 0.84239 | 0.81727 | 0.33929 | 0.57795 | 0.42757 | 0.42205 | 0.88525 | 0.33449 | 0.62242 |
| E6 | 0.84398 | 0.80235 | 0.33779 | 0.52852 | 0.41216 | 0.34981 | 0.89436 | 0.30604 | 0.60725 |
| E7 | 0.78839 | 0.75988 | 0.28030 | 0.71673 | 0.40299 | 0.55323 | 0.81237 | 0.32747 | 0.58144 |


## Uncertainty

| metric | lower | upper |
| --- | --- | --- |
| accuracy | -0.06049 | -0.03932 |
| macro_f1 | -0.07023 | -0.04748 |
| change_precision | -0.12833 | -0.02553 |
| change_recall | 0.23977 | 0.35184 |
| change_f1 | 0.05112 | 0.13164 |
| exact_destination | 0.19622 | 0.29495 |
| stable_accuracy | -0.09382 | -0.06583 |
| high_f1 | 0.21319 | 0.32310 |


## Limitations

Only one target station, one historical city dataset and one fixed chronological holdout are studied. The test set has already been used in previous project phases and the preserved legacy Phase 7 attempt; it is not a newly collected untouched external test set. Validation optimization over a finite grid can itself overfit. Intervals condition on trained models and the fixed selection and do not include training uncertainty. Calendar-day blocks may not capture multiday pollution episodes. No external station or prospective deployment test was performed.

The selected balanced logistic classifier overestimates change probabilities relative to observed frequency. No extra calibrator was fitted; Brier, log loss and ECE worsen relative to F3. This was not repaired by choosing a new configuration after inspecting test scores. A separately preregistered chronological out-of-fold calibration experiment is future work. The existing nested-calibration directory is a separate historical experiment whose lack of saved row-level predictions prevents independent recomputation here.

The original F3 destination distribution has eta=0, so destinations remain Markov-based. Detection can improve while destination errors remain. Physical histories include training-policy imputations, with flags exported. The fractions of rising stations reflect measured/imputed rates, not verified causal atmospheric transport. Hardware timing is a single warm local run for small new models, not an end-to-end deployment benchmark. Runtime and storage for unavailable historical measurements are N/A. The professor's authoritative original P2-P7 sheet is not present in the supplied local evidence, so compliance is mapped to the working descriptions only. No original 3,168-feature SVM run is fabricated.

## Supporting evidence

Use FULL_PROJECT_MATHEMATICAL_SUMMARY.md for derivations, P2_P7_COMPLIANCE_AUDIT.md for code mapping, the figure index for all 20 comparisons, and the separate independent audit for calculations.
