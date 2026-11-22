# IEEE Required File Transfer Checklist

This checklist identifies every repository file and package required by the downstream AI assistant to write the final IEEE-format university research paper without guessing implementation details or fabricating results.

---

## Part 1: Primary Package Archives (Mandatory Complete Bundles)

| Package Archive | Size (Bytes) | Member Count | Verification | Purpose |
|---|---|:---:|:---:|---|
| Phase7_Transition_Precursor_Results.zip | 11,620,686 | 208 | PASS (CRC verified) | Complete final precursor implementation, trained logistic models, 16 raw prediction CSVs, 20 publication figures, 6 vector diagrams, and research reports. |
| Phase7_Independent_Verification_Results.zip | 84,611 | 47 | PASS (CRC verified) | Complete independent numerical verification suite, recomputed metric tables, bootstrap confidence intervals, and audit matrices. |
| IEEE_Paper_Handover_Package.zip | ~150,000 | 6 | PASS (Generated) | Complete technical handover documentation, figure manifest, claims matrix, section content map, and unresolved issues audit. |

---

## Part 2: File Transfer Priority Classification

### Priority A — Absolutely Required for IEEE Paper Generation
*These files provide the primary text, tables, figures, and direct empirical evidence needed for all 11 IEEE paper sections.*

1. **Core Synthesized Research Reports:**
   - Phase7_Transition_Precursor_Results/FINAL_RESULTS_SUMMARY.md: Locked model choices, macro metrics, and trade-off summary.
   - Phase7_Transition_Precursor_Results/reports/IEEE_PAPER_CONTENT_PACKAGE.md: Pre-drafted 11-section text with equations and citations.
   - Phase7_Transition_Precursor_Results/reports/FULL_PROJECT_FINAL_REPORT.md: Comprehensive end-to-end multi-phase project narrative.
   - Phase7_Transition_Precursor_Results/reports/FULL_PROJECT_MATHEMATICAL_SUMMARY.md: Rigorous derivations for SVD, HoSVD, SVM, RFF, Markov, and Fusion.
   - Phase7_Transition_Precursor_Results/reports/PHASE6_TO_PHASE7_RECONCILIATION.md: Scope A vs B reconciliation and baseline alignment.
   - Phase7_Transition_Precursor_Results/reports/FINAL_PROJECT_READINESS.md: Packaging audit and compliance summary.

2. **Core Numerical Result CSVs:**
   - Phase7_Transition_Precursor_Results/results/final_evaluation.csv: Complete E0–E7 evaluation across validation/test and Scope A/B.
   - Phase7_Transition_Precursor_Results/results/bootstrap_confidence_intervals.csv: 1,000 block bootstrap CIs and paired contrasts vs E0 and E2.
   - Phase7_Transition_Precursor_Results/results/cross_phase_comparison.csv: Direct multi-phase comparison table on matching samples.
   - Phase7_Transition_Precursor_Results/results/transition_metrics.csv: Dedicated TP, FP, FN, TN, precision, recall, and destination metrics.
   - Phase7_Transition_Precursor_Results/results/confusion_matrices.csv: Exact 3x3 confusion matrices for all evaluated models.
   - Phase7_Transition_Precursor_Results/results/high_entry_metrics.csv: Event-specific metrics for transitions into High pollution.
   - Phase7_Transition_Precursor_Results/results/computational_cost.csv: Training/inference latency and parameter storage footprint.
   - Phase7_Transition_Precursor_Results/config/selected_configuration.json: Validation-locked winning hyperparameters.

3. **Essential Publication Figures & Architectural Diagrams:**
   - Phase7_Transition_Precursor_Results/figures/01_accuracy_and_macro_f1.png (and .pdf, data/01_*.csv)
   - Phase7_Transition_Precursor_Results/figures/02_change_recall_across_models.png (and .pdf, data/02_*.csv)
   - Phase7_Transition_Precursor_Results/figures/03_correct_destination_on_actual_transitions.png (and .pdf, data/03_*.csv)
   - Phase7_Transition_Precursor_Results/figures/04_stability_versus_transition_recall.png (and .pdf, data/04_*.csv)
   - Phase7_Transition_Precursor_Results/figures/05_transition_precision_recall_and_f1.png (and .pdf, data/05_*.csv)
   - Phase7_Transition_Precursor_Results/figures/06_selected_phase_7_versus_phase_6.png (and .pdf, data/06_*.csv)
   - Phase7_Transition_Precursor_Results/figures/10_controlled_e0_to_e7_ablations.png (and .pdf, data/10_*.csv)
   - Phase7_Transition_Precursor_Results/figures/19_transitions_into_high_pollution.png (and .pdf, data/19_*.csv)
   - Phase7_Transition_Precursor_Results/figures/20_research_progression_on_matching_test_rows.png (and .pdf, data/20_*.csv)
   - Phase7_Transition_Precursor_Results/diagrams/01_project.png (and .pdf, .json)
   - Phase7_Transition_Precursor_Results/diagrams/02_phase7.png (and .pdf, .json)
   - Phase7_Transition_Precursor_Results/diagrams/06_contribution.png (and .pdf, .json)

4. **Literature & Novelty Matrix:**
   - Phase7_Transition_Precursor_Results/literature/verified_reference_matrix.csv: 15 verified references with venue, DOI, and relevance.

---

### Priority B — Required for Mathematical and Numerical Audit Verification
*These files provide the independent calculations, prior phase anchors, and raw predictions required to prove scientific reproducibility.*

1. **Independent Audit Artifacts:**
   - Phase7_Independent_Verification_Results/results/final_verification_matrix.csv: Full 168-item scorecard (163 PASS, 4 WARNING, 3 NOT VERIFIED).
   - Phase7_Independent_Verification_Results/results/final_metrics_reconciliation.csv: Independently recalculated metrics.
   - Phase7_Independent_Verification_Results/results/graph_verification_matrix.csv: 20/20 numerical plot checks.
   - Phase7_Independent_Verification_Results/results/artifact_integrity.csv: SHA-256 hashes of all 378 frozen historical files.
   - Phase7_Independent_Verification_Results/reports/MATHEMATICAL_VERIFICATION.md
   - Phase7_Independent_Verification_Results/reports/DATA_LEAKAGE_AND_REPRODUCIBILITY.md
   - Phase7_Independent_Verification_Results/reports/ISSUES_AND_REQUIRED_CORRECTIONS.md

2. **Raw Model Predictions (Row-Level Verification):**
   - Phase7_Transition_Precursor_Results/predictions/test/E0.csv through E7.csv (8 test prediction CSVs with probabilities and decisions).
   - Phase7_Transition_Precursor_Results/predictions/validation/E0.csv through E7.csv (8 validation prediction CSVs).

3. **Historical Phase Anchor Files (Phases 2–6):**
   - Phase 2/phase2_output/phase2_summary.json: Ground-truth split boundaries, missing counts, normalization stats.
   - Air_Pollution_Phase3_Results/phase3_output/markov_transition_matrix.csv: Empirical 3x3 transition matrix.
   - phase4_output/pca_hosvd_comparison.csv: Exact reconstruction errors and compression ratios.
   - phase4_output/compression_storage.csv: Byte sizes and runtime for PCA vs HoSVD.
   - Phase5_SVM_Kernel_RFF_Results/audit_summary.json: Best validation models for linear, RBF, and RFF SVMs.
   - Phase6_Markov_SVM_Fusion_Results/config/selected_configuration.json: Validation-selected F1, F2, F3 parameters.
   - Phase6_Markov_SVM_Fusion_Results/results/final_evaluation.csv: Original Phase 6 evaluation table.
   - Phase7_Nested_CV_Calibration/results/phase7_comparison.csv: Historical nested CV comparison table.

---

### Priority C — Optional Supporting Figures and Presentation Material
*Useful for viva defense, presentation slide generation, or extended appendix material.*

- Phase7_Transition_Precursor_Results/reports/PRESENTATION_AND_VIVA_GUIDE.md: 15-slide presentation notes and 18 viva Q&As.
- Phase7_Transition_Precursor_Results/figures/07_threshold_proximity_and_missed_transitions.png
- Phase7_Transition_Precursor_Results/figures/08_target_station_rates_before_transitions.png
- Phase7_Transition_Precursor_Results/figures/09_other_station_rates_before_transitions.png
- Phase7_Transition_Precursor_Results/figures/11_validation_fusion_weight_sensitivity.png
- Phase7_Transition_Precursor_Results/figures/12_validation_decision_threshold_sensitivity.png
- Phase7_Transition_Precursor_Results/figures/13_three_state_confusion_matrices.png
- Phase7_Transition_Precursor_Results/figures/14_probability_reliability.png
- Phase7_Transition_Precursor_Results/figures/15_additional_model_computation_and_storage.png
- Phase7_Transition_Precursor_Results/figures/16_pca_and_hosvd_reconstruction.png
- Phase7_Transition_Precursor_Results/figures/17_compression_dimension_and_classification.png
- Phase7_Transition_Precursor_Results/figures/18_exact_rbf_versus_revised_rff.png
- Phase7_Transition_Precursor_Results/diagrams/03_markov.png
- Phase7_Transition_Precursor_Results/diagrams/04_compression.png
- Phase7_Transition_Precursor_Results/diagrams/05_decision.png
