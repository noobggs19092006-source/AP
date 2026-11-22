# IEEE Paper Section-by-Section Content Map

This document establishes the precise technical blueprint for all 11 mandatory sections of our university mathematics research paper in IEEE format. For each section, it maps exact objectives, mathematical formulations, verified figures, evidence tables, safe claims, and claims to avoid.

---

## Section I: Abstract
- **Main Objective:** Provide a self-contained, quantitative summary of the research problem, methodology, key findings, and honest scientific trade-off.
- **Required Topics:**
  - Motivation: High overall accuracy in air-pollution models often masks complete failure to predict abrupt state changes.
  - Method: Transition-Precursor-Aware Markov–SVM Fusion integrating threshold proximity, target temporal rates, and cross-station spatial summaries with Phase 6 F3 change/destination framework.
  - Dataset: 35,064 hourly records across 12 Beijing monitoring stations (2013–2017), predicting next-hour pollution category (Low <35, Moderate 35–75, High >=75 ug/m3).
  - Empirical Findings: Change recall increases from 0.00% (Markov) and 26.43% (Phase 6 F3) to 55.89% (Selected Phase 7 Model E4), and exact destination accuracy increases from 15.59% to 40.11%.
  - Disclosed Trade-off: False alarms increase from 212 to 622, overall accuracy declines from 88.46% to 83.52%, and stable-state accuracy declines from 95.89% to 87.94%.
- **Key Equations:** $q_t^{new} = \alpha q_t^P + (1-\alpha) q_t^{F3}$, $J = 0.5 \text{MacroF1} + 0.5 \text{ChangeF1}$.
- **Safe Claims:** Proximity and trend precursors substantially improve transition detection sensitivity and destination identification.
- **Claims to Avoid:** Do not claim superior overall accuracy; do not claim universal Pareto dominance.

---

## Section II: Introduction — Research Problem, Gap and Contributions
- **Main Objective:** Define the air-quality transition prediction problem, identify the persistence blind spot in existing Markov/SVM models, and state explicit research contributions.
- **Required Topics:**
  - Severe societal and health impacts of sudden particulate matter (PM2.5) pollution spikes.
  - Class imbalance in environmental monitoring: 90.75% of hourly intervals are stable, while only 9.25% exhibit state changes.
  - Research Gap: First-order Markov chains maximize persistence accuracy (100%) but predict zero transitions in hard predictions. Standalone SVMs detect transitions (46.39%) but yield lower overall accuracy (83.69%). Phase 6 fusion achieved 26.43% recall without early-warning signals.
  - Central Research Question: Can threshold proximity and short-term temporal/spatial trends improve next-hour state change detection when fused with Markov–SVM probabilities?
  - Explicit Contributions:
    1. Formulation of compact, physically interpretable precursor feature groups (threshold distances, local velocity, cross-station rise fraction).
    2. Regularized binary change probability modeling coupled with the Phase 6 F3 destination distribution.
    3. Rigorous validation-locked experimental protocol (E0–E7) with 1,000 paired block bootstrap uncertainty quantification.
- **Recommended Figures & Diagrams:**
  - `diagrams/01_project.png`: End-to-end multi-phase project architecture.
  - `diagrams/06_contribution.png`: Transition blind spot and precursor fusion concept.
- **Supporting Paths:** `Phase7_Transition_Precursor_Results/reports/PHASE7_RESEARCH_REPORT.md`.

---

## Section III: Mathematical Background
- **Main Objective:** Provide complete, formal mathematical formulations for matrix SVD/PCA, tensor HoSVD, linear/kernel SVM, RFF, and Markov chains.
- **Required Topics & Equations:**
  - **SVD & PCA:** Flattened tensor matrix $X \in \mathbb{R}^{N \times D}$ ($D = 3168$). Covariance/Gram matrix SVD: $X = U \Sigma V^T$. Subspace projection: $z = V_k^T (x - \mu)$. Retained energy: $E_k = \sum_{i=1}^k \sigma_i^2 / \sum_{i=1}^D \sigma_i^2$.
  - **HoSVD:** 3rd-order tensor $\mathcal{X} \in \mathbb{R}^{24 \times 12 \times 11}$. Mode-$n$ unfolding matrices $X_{(1)}, X_{(2)}, X_{(3)}$. Factor matrices $U^{(1)} \in \mathbb{R}^{24 \times r_1}, U^{(2)} \in \mathbb{R}^{12 \times r_2}, U^{(3)} \in \mathbb{R}^{11 \times r_3}$. Core tensor: $\mathcal{S} = \mathcal{X} \times_1 U^{(1)T} \times_2 U^{(2)T} \times_3 U^{(3)T}$.
  - **SVM & Regularization:** Soft-margin objective $\min_{w,b,\xi} \frac{1}{2}\|w\|^2 + C \sum_i \xi_i$ s.t. $y_i(w^T x_i + b) \ge 1 - \xi_i$.
  - **Kernel SVM & RFF:** RBF kernel $K(x,x') = \exp(-\gamma \|x-x'\|^2)$. Bochner theorem and Random Fourier Feature mapping $z(x) = \sqrt{2/D}[\cos(w_1^T x + b_1), \dots, \sin(w_D^T x + b_D)]$.
  - **Markov Chain:** Row-stochastic transition matrix $P_{ij} = P(S_{t+1}=j \mid S_t=i) = N_{ij} / \sum_k N_{ik}$.
- **Recommended Figures & Diagrams:**
  - `diagrams/03_markov.png`: 3-state Markov chain transition diagram.
  - `diagrams/04_compression.png`: Matrix PCA unfolding vs 3-mode tensor HoSVD truncation.
  - `figures/16_pca_and_hosvd_reconstruction.png`: Reconstruction error vs dimension.
- **Supporting Paths:** `Phase7_Transition_Precursor_Results/reports/FULL_PROJECT_MATHEMATICAL_SUMMARY.md`.

---

## Section IV: Existing Methods
- **Main Objective:** Review baseline models from Phases 3–6, detailing their architectures, strengths, and failure modes.
- **Required Topics:**
  - Phase 3 Markov Baseline (E0): High overall accuracy (90.75%), high persistence accuracy (100%), 0% change recall.
  - Phase 5 Standalone SVMs (E1): Linear SVM on PCA-C ($k=576$), Exact RBF SVM on PCA-B ($k=192$, 4,033 support vectors), RFF Linear SVM ($D=2048$).
  - Phase 6 Markov–SVM Fusion:
    - F1 Fixed: $p_t^{F1} = \lambda p_t^{SVM} + (1-\lambda) p_t^{Markov}$.
    - F2 Adaptive: Confidence thresholding on $\tau$.
    - F3 Transition/Destination Blending: Change probability $q_t^{F3} = \rho(1 - p_t^{SVM}(S_t)) + (1-\rho)(1 - P_{S_t, S_t})$ and conditional destination $d_t^{F3}(j) = \frac{\eta p_t^{SVM}(j) + (1-\eta) P_{S_t, j}}{\sum_{k \ne S_t}[\eta p_t^{SVM}(k) + (1-\eta) P_{S_t, k}]}$.
- **Recommended Figures:**
  - `figures/17_compression_dimension_and_classification.png`: Compression dimension vs SVM macro-F1.
  - `figures/18_exact_rbf_versus_revised_rff.png`: Exact RBF vs RFF comparison.
- **Supporting Paths:** `Phase6_Markov_SVM_Fusion_Results/results/final_evaluation.csv`.

---

## Section V: Proposed Method — Transition-Precursor-Aware Fusion
- **Main Objective:** Provide exact mathematical definitions and algorithmic workflow for the new Phase 7 precursor feature extraction and probability fusion.
- **Required Topics & Equations:**
  - Physical PM2.5 Reconstruction: $m_{s,t} = z_{s,t} \cdot \sigma_s + \mu_s$ using training-derived statistics.
  - Feature Group A (Threshold Proximity): $d_{35,t} = |m_{0,t} - 35|$, $d_{75,t} = |m_{0,t} - 75|$, $d_{min,t} = \min(d_{35,t}, d_{75,t})$, plus current state one-hot indicators.
  - Feature Group B (Target Temporal Trends): $v_{3,t} = \frac{m_{0,t} - m_{0,t-3}}{3}$, $v_{6,t} = \frac{m_{0,t} - m_{0,t-6}}{6}$.
  - Feature Group C (Cross-Station Spatial Trends): $v_{s,t} = \frac{m_{s,t} - m_{s,t-3}}{3}$ for $s=1,\dots,11$. Summaries: $v_{mean,t} = \frac{1}{11}\sum_{s=1}^{11} v_{s,t}$, $v_{max,t} = \max_{s \ne 0} v_{s,t}$, $f_{rise,t} = \frac{1}{11}\sum_{s=1}^{11} \mathbf{1}[v_{s,t} > 0]$.
  - Precursor Logistic Classifier: $q_t^P = \sigma(w^T h_t + b) = \frac{1}{1 + e^{-(w^T h_t + b)}}$.
  - Two-Stage Fusion Formulation: $q_t^{new} = \alpha q_t^P + (1-\alpha) q_t^{F3}$.
  - Destination Probability Reconstruction: $p_t^{new}(S_t) = 1 - q_t^{new}$, and $p_t^{new}(j) = q_t^{new} \cdot d_t^{F3}(j)$ for $j \ne S_t$.
  - Scope B Deterministic Fallback: When $S_t = -1$, fallback to calibrated feature-only SVM probabilities.
- **Recommended Figures & Diagrams:**
  - `diagrams/02_phase7.png`: Detailed Phase 7 architecture and signal flow.
  - `diagrams/05_decision.png`: Transition prediction and fallback flowchart.
  - `figures/07_threshold_proximity_and_missed_transitions.png`: Threshold distance distribution.
  - `figures/08_target_station_rates_before_transitions.png`: Rate of change velocity before state changes.
- **Supporting Paths:** `Phase7_Transition_Precursor_Results/source/physical_precursors.py`, `precursor_fusion.py`.

---

## Section VI: Mathematical Analysis
- **Main Objective:** Prove theoretical properties of the precursor fusion model (simplex normalization, destination bounds, accuracy decomposition).
- **Required Topics & Equations:**
  - **Probability Simplex Preservation:** Proof that $\sum_{j=0}^2 p_t^{new}(j) = 1$ and $p_t^{new}(j) \ge 0$ whenever $q_t^{new} \in [0, 1]$ and $\sum_{j \ne i} d_t^{F3}(j) = 1$.
  - **Weighted Accuracy Decomposition:** Scope A accuracy identity:
    $$\text{Accuracy} = (1 - \pi_{change}) \cdot A_{stable} + \pi_{change} \cdot A_{change}$$
    where $\pi_{change} = 526 / 5685 = 0.092524$. Checked on E4: $(0.907476)(0.879434) + (0.092524)(0.401141) = 0.835180$ (exact match).
  - **Destination Accuracy Upper Bound:** $A_{change} \le \text{Recall}_{change}$ (an incorrect transition prediction cannot yield a correct destination).
  - **Selection Objective:** $J = 0.5 \text{MacroF1} + 0.5 \text{ChangeF1}$.
- **Supporting Paths:** `Phase7_Transition_Precursor_Results/reports/FULL_PROJECT_MATHEMATICAL_SUMMARY.md`.

---

## Section VII: Experimental Design
- **Main Objective:** Define chronological splits, feature scaling, grid search, validation-locking protocol, and evaluation scopes.
- **Required Topics:**
  - Chronological Partitioning: Train (2013-03-01 to 2015-12-31, N=24,108), Validation (2016-01-01 to 2016-06-30, N=4,239), Test (2016-07-01 to 2017-02-28, N=5,720).
  - Scope A (N=5,685 test) vs Scope B (N=5,720 test, 35 missing current states).
  - Controlled Experiments E0–E7: E0 (Markov), E1 (Linear SVM), E2 (Phase 6 F3), E3 (F3 + Proximity), E4 (F3 + Proximity + Target Trend), E5 (F3 + All 3 Groups), E6 (E5 + Decision Theta), E7 (Precursor Only alpha=1).
  - Hyperparameter Search: $C \in \{0.1, 1.0, 10.0\}$, class weight $\in \{\text{None}, \text{balanced}\}$, $\alpha \in \{0, 0.25, 0.5, 0.75, 1.0\}$, $\theta \in \{0.2, 0.3, 0.4, 0.5, 0.6, 0.7\}$.
  - Validation Lock: Selection locked on Validation Scope A before test set evaluation. Winning model: E4 (temporal_C1.0_balanced, alpha=0.75, J=0.6202).
- **Recommended Figures:**
  - `figures/11_validation_fusion_weight_sensitivity.png`: Alpha sensitivity curve.
  - `figures/12_validation_decision_threshold_sensitivity.png`: Theta sensitivity curve.
- **Supporting Paths:** `Phase7_Transition_Precursor_Results/config/selected_configuration.json`.

---

## Section VIII: Results
- **Main Objective:** Present empirical findings across all controlled experiments on the hold-out test set.
- **Required Topics & Tables:**
  - Comprehensive Test Scope A Table (E0 through E7): Accuracy, Macro-F1, Change Precision, Change Recall, Change-F1, Destination Accuracy, Stable Accuracy, High-Entry F1.
  - Ablation Analysis: E3 improves recall from 26.43% to 55.70%. E4 adds target rates to achieve Change-F1 0.4078. E5 test recall reaches 57.79%.
  - High-Pollution Entry Results: E4 achieves High-entry F1 of 0.3026 (recall 21.3%) vs 0.0344 (4.9%) for F3 and 0.0000 for Markov.
  - Confusion Matrix Analysis: E4 detects 294 true transitions (TP), misses 232 (FN), generates 622 false alarms (FP).
- **Recommended Figures:**
  - `figures/01_accuracy_and_macro_f1.png`: Accuracy and Macro-F1 across models.
  - `figures/02_change_recall_across_models.png`: Change recall progression.
  - `figures/03_correct_destination_on_actual_transitions.png`: Destination accuracy on actual changes.
  - `figures/05_transition_precision_recall_and_f1.png`: Transition detection PRF.
  - `figures/10_controlled_e0_to_e7_ablations.png`: Controlled ablation bar chart.
  - `figures/13_three_state_confusion_matrices.png`: Confusion matrices.
  - `figures/19_transitions_into_high_pollution.png`: High-entry performance.
- **Supporting Paths:** `Phase7_Transition_Precursor_Results/results/final_evaluation.csv`.

---

## Section IX: Comparison and Statistical Analysis
- **Main Objective:** Rigorously contrast Phase 7 E4 against Phase 6 F3 and baselines using paired block bootstrap confidence intervals and Pareto analysis.
- **Required Topics:**
  - Paired Block Bootstrap: 1,000 calendar-day (24-hour block) resamples.
    - $\Delta \text{Change Recall} = +29.46\text{ pp}$ ($95\%\text{ CI: }[+23.98, +35.18]$) — Statistically significant gain.
    - $\Delta \text{Change-F1} = +0.0908$ ($95\%\text{ CI: }[+0.0511, +0.1316]$) — Statistically significant gain.
    - $\Delta \text{Exact Destination} = +24.52\text{ pp}$ ($95\%\text{ CI: }[+19.62, +29.50]$) — Statistically significant gain.
    - $\Delta \text{Accuracy} = -4.94\text{ pp}$ ($95\%\text{ CI: }[-6.05, -3.93]$) — Statistically significant cost.
    - $\Delta \text{Stable Accuracy} = -7.95\text{ pp}$ ($95\%\text{ CI: }[-9.38, -6.58]$) — Statistically significant cost.
  - Computational Cost: Feature extraction latency 0.18 ms/sample (5.5 kHz throughput). Model memory 14.4 KB.
- **Recommended Figures:**
  - `figures/04_stability_versus_transition_recall.png`: Pareto curve of stability vs recall.
  - `figures/06_selected_phase_7_versus_phase_6.png`: Direct paired contrast with error bars.
  - `figures/15_additional_model_computation_and_storage.png`: Computation and memory footprint.
  - `figures/20_research_progression_on_matching_test_rows.png`: Multi-phase research progression.
- **Supporting Paths:** `Phase7_Transition_Precursor_Results/results/bootstrap_confidence_intervals.csv`.

---

## Section X: Limitations and Scientific Honesty
- **Main Objective:** Transparently disclose all technical qualifications, performance regressions, and methodological boundaries.
- **Required Topics:**
  - **False Alarm Increase:** Detecting 155 additional transitions incurs 410 additional false alarms (precision drops from 39.60% to 32.10%).
  - **Calibration Degradation:** Brier score increases from 0.1814 to 0.2372; ECE increases from 0.0570 to 0.0682. Balanced logistic weighting stretches probabilities toward extremes.
  - **Single Target Station:** Evaluation is localized to Aotizhongxin; cross-city generalization remains unverified.
  - **Cross-Station Causality Caution:** Other monitoring stations reflect correlated regional air masses rather than proven causal meteorological advection.
  - **Historical Nested CV Discrepancy:** The separate `Phase7_Nested_CV_Calibration` experiment used Scope B rows and in-sample calibrator training; it is not directly comparable to Scope A benchmarks.
- **Recommended Figures:**
  - `figures/14_probability_reliability.png`: Reliability diagrams and calibration gaps.
- **Supporting Paths:** `Phase7_Transition_Precursor_Results/reports/PHASE7_LIMITATIONS.md`.

---

## Section XI: Conclusion and Future Work
- **Main Objective:** Summarize verified findings, provide concluding assessment of the precursor-aware fusion framework, and suggest future research avenues.
- **Required Topics:**
  - Concluding Assessment: The proposed Transition-Precursor-Aware Markov–SVM Fusion successfully resolves the Markov transition blind spot, doubling transition recall (55.89% vs 26.43%) and more than doubling destination accuracy (40.11% vs 15.59%).
  - Practical Value: For environmental health and public alert systems, detecting over half of pollution state changes (and 21.3% of severe High pollution entries) outweighs the modest 4.94 pp decrease in overall accuracy.
  - Future Work: Incorporating explicitly measured wind speed/direction vectors, exploring cost-sensitive asymmetric loss functions to suppress false alarms, and extending to multi-station simultaneous transition prediction.
