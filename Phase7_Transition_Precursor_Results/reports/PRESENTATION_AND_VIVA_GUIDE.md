# Fifteen-slide presentation and viva guide

Target duration: 20 minutes.

| slide | minutes | speaker_notes |
| --- | --- | --- |
| Title | 0.50000 | State the project title and the precise next-hour state-change objective. Do not invent author or institution details. |
| Introduction | 1.50000 | Explain why high persistence accuracy can miss every transition. Use E0 count: 5,159 stable versus 526 changed. |
| Literature review | 1.50000 | Contrast Bekkar spatial-temporal concentration forecasting with Zakaria Markov categories. Cite sources 1-2 and the targeted gap. |
| Objectives | 1.00000 | Detect more changes while measuring false alarms, correct destination and stability. |
| Mathematical theory | 2.00000 | Explain tensor as a three-axis array; PCA as orthogonal projection; HoSVD as three mode projections; margin as distance to a separating boundary. Use diagram 4. |
| Proposed methodology | 2.00000 | Use diagram 2. Explain d35/d75/dmin, v3/v6, other-station summaries, logistic risk and alpha. |
| Dataset and preprocessing | 1.50000 | Twelve stations, hourly data, 24-hour windows, training-only normalization, chronological splits and missing-state fallback. |
| Algorithm | 1.50000 | Walk through train-only fitting, validation selection lock, then one test evaluation; use diagram 5. |
| Experimental setup | 1.50000 | Explain E0-E7, C grid, weighting, alpha, theta, Scope A/B, and 1,000 paired day blocks. |
| Results | 2.00000 | Show figures 1, 5 and 6; state recall gain and accuracy loss together. |
| Comparison | 1.50000 | Show figure 10 and figure 20. E5 is an ablation, not a test-reselected winner. |
| Applications | 0.75000 | Potential research alert prioritization, not a certified health-warning system; false alarms have practical costs. |
| Limitations and future scope | 1.00000 | Discuss miscalibration, shared historical test use, fixed destinations and external validation. |
| Conclusion | 0.75000 | Repeat the measured trade-off and the validation-selected method. |
| References | 0.50000 | Use the fifteen-paper reference matrix; preserve access-depth caveats. |


## Main questions and answers

## 1. Why is Markov accuracy high?

5,159 of 5,685 observed-current test outcomes remain in the same state. The diagonal is the maximum in each saved transition row, so hard predictions are persistence.

## 2. Why does Markov miss changes?

It uses only the discrete current state and always selects that state. Physical proximity and trends do not enter its row selection.

## 3. How many changes does standalone SVM detect?

The Phase 6-compatible calibrated linear SVM detects 244 of 526. Historical raw hard SVM decisions are separate in cross_phase_comparison.csv.

## 4. Does Phase 6 improve the trade-off?

F3 restores stability and accuracy relative to the standalone calibrated SVM but lowers change recall. A trade-off is not dominance.

## 5. Do threshold features help?

E3 test change F1 is 0.4058, versus F3 0.3170; stability declines.

## 6. Do target trends help?

E4 test change F1 is 0.4078, a small increase over E3; validation selects E4.

## 7. Do cross-station features help?

E5 test change F1 is 0.4276, but its validation objective is below E4. General superiority is not established.

## 8. Does the full model beat individual groups?

On the test trade-off E5 is stronger than E3/E4, but not by the locked validation criterion; no test-based reselection was made.

## 9. Are there more false alarms?

Yes: 622 for E4 versus 212 for F3. Whether that is acceptable requires an application-specific false-alarm cost.

## 10. Does exact destination improve?

Yes, from 0.1559 to 0.4011 on actual changes.

## 11. Does stable accuracy decline?

Yes, from 0.9589 to 0.8794.

## 12. Does overall accuracy improve?

No: 0.8846 to 0.8352.

## 13. Does macro-F1 improve?

No: 0.8661 to 0.8074.

## 14. Does calibration improve?

No. Brier, log loss and confidence ECE worsen. Balanced class weighting is a relevant mechanism; probabilities are not claimed calibrated.

## 15. Is there statistical support?

Paired day-block intervals exclude zero for increased recall/change-F1 and for decreased accuracy/stability. This supports both sides of the trade-off under the bootstrap assumptions.

## 16. Is computation justified?

The added logistic models are small and fast; computational cost is modest. Operational justification remains contingent on alarm costs and external validation.

## 17. What should the paper highlight?

Highlight the validation-selected E4 as the proposed method, F3 and Markov as competing trade-offs, and E5 as a prespecified ablation with promising test evidence. Do not claim a universally best model.

## 18. What remains limited?

Single target, historical reuse, calibration deterioration, fixed Markov destination, one holdout, short bootstrap blocks and incomplete original professor requirements.

## Backup calculations

| model | accuracy | macro_f1 | change_precision | change_recall | change_f1 | exact_destination | stable_accuracy | high_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | 0.90748 | 0.88951 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 1.00000 | 0.00000 |
| E1 | 0.83694 | 0.80233 | 0.28339 | 0.46388 | 0.35184 | 0.41065 | 0.88040 | 0.30493 |
| E2 | 0.88461 | 0.86615 | 0.39601 | 0.26426 | 0.31699 | 0.15589 | 0.95891 | 0.03436 |
| E4 | 0.83518 | 0.80742 | 0.32096 | 0.55894 | 0.40777 | 0.40114 | 0.87943 | 0.30263 |

For E4, change precision = 294/(294+622)=0.320961; recall=294/526=0.558935. Correct destination is 211/526=0.401141. Accuracy=(526*A_change+5159*A_stable)/5685=0.835180.

A probability is a number from zero to one; calibration asks whether predicted frequencies agree with observed frequencies. A confidence interval reflects repeated block-resampling variability, not a guarantee about a future city. Alpha mixes two change-risk estimates; theta is a hard-decision cutoff. They solve different problems. The most likely class need not change at q=.5 because change mass is split between two destinations.

## Graph paths and captions

Figure 1. `Phase7_Transition_Precursor_Results/figures/01_accuracy_and_macro_f1.png`

Suggested caption: Accuracy and macro F1; chronological holdout, project-defined states. Numerical data: `Phase7_Transition_Precursor_Results/figures/data/01_accuracy_and_macro_f1.csv`. Supports: Accuracy and macro F1.

Figure 2. `Phase7_Transition_Precursor_Results/figures/02_change_recall_across_models.png`

Suggested caption: Change recall across models; chronological holdout, project-defined states. Numerical data: `Phase7_Transition_Precursor_Results/figures/data/02_change_recall_across_models.csv`. Supports: Change recall across models.

Figure 3. `Phase7_Transition_Precursor_Results/figures/03_correct_destination_on_actual_transitions.png`

Suggested caption: Correct destination on actual transitions; chronological holdout, project-defined states. Numerical data: `Phase7_Transition_Precursor_Results/figures/data/03_correct_destination_on_actual_transitions.csv`. Supports: Correct destination on actual transitions.

Figure 4. `Phase7_Transition_Precursor_Results/figures/04_stability_versus_transition_recall.png`

Suggested caption: Stability versus transition recall; chronological holdout, project-defined states. Numerical data: `Phase7_Transition_Precursor_Results/figures/data/04_stability_versus_transition_recall.csv`. Supports: Stability versus transition recall.

Figure 5. `Phase7_Transition_Precursor_Results/figures/05_transition_precision_recall_and_f1.png`

Suggested caption: Transition precision recall and F1; chronological holdout, project-defined states. Numerical data: `Phase7_Transition_Precursor_Results/figures/data/05_transition_precision_recall_and_f1.csv`. Supports: Transition precision recall and F1.

Figure 6. `Phase7_Transition_Precursor_Results/figures/06_selected_phase_7_versus_phase_6.png`

Suggested caption: Selected Phase 7 versus Phase 6; chronological holdout, project-defined states. Numerical data: `Phase7_Transition_Precursor_Results/figures/data/06_selected_phase_7_versus_phase_6.csv`. Supports: Selected Phase 7 versus Phase 6.

Figure 7. `Phase7_Transition_Precursor_Results/figures/07_threshold_proximity_and_missed_transitions.png`

Suggested caption: Threshold proximity and missed transitions; chronological holdout, project-defined states. Numerical data: `Phase7_Transition_Precursor_Results/figures/data/07_threshold_proximity_and_missed_transitions.csv`. Supports: Threshold proximity and missed transitions.

Figure 8. `Phase7_Transition_Precursor_Results/figures/08_target_station_rates_before_transitions.png`

Suggested caption: Target station rates before transitions; chronological holdout, project-defined states. Numerical data: `Phase7_Transition_Precursor_Results/figures/data/08_target_station_rates_before_transitions.csv`. Supports: Target station rates before transitions.

Figure 9. `Phase7_Transition_Precursor_Results/figures/09_other_station_rates_before_transitions.png`

Suggested caption: Other station rates before transitions; chronological holdout, project-defined states. Numerical data: `Phase7_Transition_Precursor_Results/figures/data/09_other_station_rates_before_transitions.csv`. Supports: Other station rates before transitions.

Figure 10. `Phase7_Transition_Precursor_Results/figures/10_controlled_e0_to_e7_ablations.png`

Suggested caption: Controlled E0 to E7 ablations; chronological holdout, project-defined states. Numerical data: `Phase7_Transition_Precursor_Results/figures/data/10_controlled_e0_to_e7_ablations.csv`. Supports: Controlled E0 to E7 ablations.

Figure 11. `Phase7_Transition_Precursor_Results/figures/11_validation_fusion_weight_sensitivity.png`

Suggested caption: Validation fusion weight sensitivity; chronological holdout, project-defined states. Numerical data: `Phase7_Transition_Precursor_Results/figures/data/11_validation_fusion_weight_sensitivity.csv`. Supports: Validation fusion weight sensitivity.

Figure 12. `Phase7_Transition_Precursor_Results/figures/12_validation_decision_threshold_sensitivity.png`

Suggested caption: Validation decision threshold sensitivity; chronological holdout, project-defined states. Numerical data: `Phase7_Transition_Precursor_Results/figures/data/12_validation_decision_threshold_sensitivity.csv`. Supports: Validation decision threshold sensitivity.

Figure 13. `Phase7_Transition_Precursor_Results/figures/13_three_state_confusion_matrices.png`

Suggested caption: Three state confusion matrices; chronological holdout, project-defined states. Numerical data: `Phase7_Transition_Precursor_Results/figures/data/13_three_state_confusion_matrices.csv`. Supports: Three state confusion matrices.

Figure 14. `Phase7_Transition_Precursor_Results/figures/14_probability_reliability.png`

Suggested caption: Probability reliability; chronological holdout, project-defined states. Numerical data: `Phase7_Transition_Precursor_Results/figures/data/14_probability_reliability.csv`. Supports: Probability reliability.

Figure 15. `Phase7_Transition_Precursor_Results/figures/15_additional_model_computation_and_storage.png`

Suggested caption: Additional model computation and storage; chronological holdout, project-defined states. Numerical data: `Phase7_Transition_Precursor_Results/figures/data/15_additional_model_computation_and_storage.csv`. Supports: Additional model computation and storage.

Figure 16. `Phase7_Transition_Precursor_Results/figures/16_pca_and_hosvd_reconstruction.png`

Suggested caption: PCA and HoSVD reconstruction; chronological holdout, project-defined states. Numerical data: `Phase7_Transition_Precursor_Results/figures/data/16_pca_and_hosvd_reconstruction.csv`. Supports: PCA and HoSVD reconstruction.

Figure 17. `Phase7_Transition_Precursor_Results/figures/17_compression_dimension_and_classification.png`

Suggested caption: Compression dimension and classification; chronological holdout, project-defined states. Numerical data: `Phase7_Transition_Precursor_Results/figures/data/17_compression_dimension_and_classification.csv`. Supports: Compression dimension and classification.

Figure 18. `Phase7_Transition_Precursor_Results/figures/18_exact_rbf_versus_revised_rff.png`

Suggested caption: Exact RBF versus revised RFF; chronological holdout, project-defined states. Numerical data: `Phase7_Transition_Precursor_Results/figures/data/18_exact_rbf_versus_revised_rff.csv`. Supports: Exact RBF versus revised RFF.

Figure 19. `Phase7_Transition_Precursor_Results/figures/19_transitions_into_high_pollution.png`

Suggested caption: Transitions into High pollution; chronological holdout, project-defined states. Numerical data: `Phase7_Transition_Precursor_Results/figures/data/19_transitions_into_high_pollution.csv`. Supports: Transitions into High pollution.

Figure 20. `Phase7_Transition_Precursor_Results/figures/20_research_progression_on_matching_test_rows.png`

Suggested caption: Research progression on matching test rows; chronological holdout, project-defined states. Numerical data: `Phase7_Transition_Precursor_Results/figures/data/20_research_progression_on_matching_test_rows.csv`. Supports: Research progression on matching test rows.
