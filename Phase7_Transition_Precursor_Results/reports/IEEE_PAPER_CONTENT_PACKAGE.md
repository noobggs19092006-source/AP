# IEEE paper content package

## Abstract

E4 detects 294 of 526 changes (55.89%) versus 139 (26.43%) for F3. False alarms increase from 212 to 622. Accuracy changes from 88.46% to 83.52%, and macro-F1 from 0.8661 to 0.8074. Correct destination accuracy increases from 15.59% to 40.11%, while stable-state accuracy declines from 95.89% to 87.94%. This is a transition-sensitivity trade-off, not an overall accuracy improvement. The experiment uses 24-hour histories at twelve Beijing monitoring stations and a validation-locked logistic change-risk fusion. Independent numerical checks and paired day-block intervals support a recall/stability trade-off.

## Introduction — Research problem, gap and contributions

Aggregate classification accuracy can conceal missed transitions when pollution states persist. Prior work includes spatial-temporal concentration forecasting [1] and Markov pollution-state forecasting [2]. Their tasks differ from this controlled next-hour category-transition problem. The contribution is an interpretable extension of a fixed SVM-Markov framework, genuine feature ablations, and auditable evidence of both benefits and costs. This targeted bibliography does not establish first-ever novelty.

## Mathematical Background



### Data, notation and states

Let t denote the final observed hour, s a monitoring station and f a numerical channel. A history tensor X_t has shape 24 by 12 by 11. Its rows run from t-23 through t; the label y_t is the state at t+1. The target station is Aotizhongxin, station index 0; PM2.5 is channel 0. A state is 0 for m<35, 1 for 35<=m<75, and 2 for m>=75, with m in micrograms per cubic metre. These instantaneous project categories are not official AQI categories. Missing observed current measurements retain state -1.

Phase 2 forward-fills no more than six hours within each chronological partition, then uses training medians. It standardizes each station-channel independently. For training mean mu_sf and standard deviation sigma_sf, z_tsf=(x_tsf-mu_sf)/sigma_sf. The inverse is x_tsf=z_tsf sigma_sf+mu_sf. Float32 storage makes inverse values approximate; the experiment checks observed historical values within 0.001 micrograms per cubic metre. Imputed histories cannot establish an observed current state. New scalers are fitted on training rows with observed current states only.

### SVD and PCA

Flatten a centered history into a row in R^3168, forming the training matrix A. Its singular value decomposition is A=U Sigma V^T, with orthonormal columns U,V and nonnegative singular values in Sigma. For retained dimension k, the feature row is a V_k and reconstruction is a V_k V_k^T. Add back the training mean to reconstruct uncentered values. The first k right singular vectors maximize captured training variance. For the truncated matrix SVD, squared Frobenius reconstruction error equals the sum of squared discarded singular values. This optimality concerns unconstrained rank-k matrix approximation and does not imply best predictive accuracy.

The implementation in AP/phase4_output/phase4_pca_hosvd.py fits training means and accumulated Gram matrices; eig_desc obtains principal axes, evaluate_split measures error, and storage_numbers separates basis and feature storage. A flattened input needs 3,168 numbers per sample; a code needs k. The logical code-only compression ratio is 3168/k. Total compression must include the mean and 3168*k basis entries, their precision, and the number of samples. On-disk compressed NPZ sizes are a different quantity.

Relative reconstruction error is ||A-A_hat||_F / ||A||_F. Mean squared error divides squared error by the number of scalar entries. Retained energy is ||A_hat||_F^2 / ||A||_F^2 for orthogonal projection. Compare methods on the same centered data and split. A reconstruction-versus-dimension curve is not a classification-versus-dimension curve; both are supplied separately.

### HoSVD and tensor compression

For a third-order tensor T, mode-n unfolding T_(n) arranges mode-n fibers as columns of a matrix. Orthonormal mode matrices U_1, U_2, U_3 span retained time, station and channel subspaces. With ranks r_1,r_2,r_3, the core G=T times_1 U_1^T times_2 U_2^T times_3 U_3^T has shape r_1 by r_2 by r_3. The symbol times_n denotes multiplication along tensor mode n; other modes retain their dimensions. Reconstruction is G times_1 U_1 times_2 U_2 times_3 U_3. Flattening the core gives k=r_1*r_2*r_3 SVM features.

The shared mode matrices in this project are fitted from aggregated training mode Gram matrices, not separately fitted on test samples. Their orthogonal projections preserve the tensor axes. The usual squared projection-error upper bound is the sum, over modes, of the discarded mode singular-value energies. For the stacked training collection this follows by sequential orthogonal projections and a telescoping decomposition: each residual term is bounded by the corresponding one-mode projection residual. A truncated HoSVD is generally not the optimal multilinear-rank approximation. Do not transfer the matrix SVD optimality theorem to arbitrary tensor rank constraints.

The stored mode basis requires 24*r_1+12*r_2+11*r_3 numbers, plus the mean and per-sample core. The code-only ratio is 3168/(r_1*r_2*r_3). A mode rank cannot exceed its original mode dimension. AP/phase4_output/hosvd_truncation_bound.csv and pca_hosvd_comparison.csv preserve historical rank/error evidence. No earlier compression model is retrained by Phase 7.

### SVM and kernels

For binary labels a_i in {-1,+1}, a linear decision function is f(x)=w^T x+b. Geometric margin is |f(x)|/||w||. The conventional soft-margin formulation minimizes (1/2)||w||^2+C sum_i xi_i subject to a_i f(x_i)>=1-xi_i and xi_i>=0. C trades margin regularization against training violations. LinearSVC uses its configured loss and multiclass strategy; its common squared-hinge objective replaces linear slack penalty by a squared penalty. The report must distinguish this implementation detail from the conventional introductory primal.

The RBF kernel K(x,x')=exp(-gamma ||x-x'||^2) measures similarity without explicitly constructing an infinite-dimensional feature vector. gamma controls locality. An exact SVC trains kernel classifiers, with internal one-versus-one multiclass handling; requesting an OvR-shaped decision output does not retrain it as one-versus-rest. LinearSVC uses one-versus-rest in this saved project model. Class order is verified as [0,1,2]. A hard margin decision and a calibrated probability argmax can differ. E1 uses the calibrated argmax used by Phase 6; historical Phase 5 hard decisions are reported separately.

### Random Fourier features

For the above RBF kernel, draw omega_j from N(0,2 gamma I) and b_j uniformly on [0,2*pi]. Define z_j(x)=sqrt(2/D) cos(omega_j^T x+b_j), j=1,...,D. The expectation of z(x)^T z(x') equals the RBF kernel. Finite D introduces approximation error; larger D usually improves approximation in expectation, not necessarily each observed classification score. Fit a linear classifier on z(x). Prediction work scales with input dimension times D plus linear classifier work; exact kernel prediction depends on support-vector count. Random seeds and mapping parameters must be preserved. The historical RFF models and revised experiments remain separate; no new random map is substituted for a saved one.

### Markov chain

Let N_ij count consecutive valid observed training-state pairs i to j separated by exactly one hour. P_ij=N_ij/sum_j N_ij. Every row sums to one and P_ij>=0. For known current state i, p_M(j)=P_ij. The hard prediction maximizes that row. When diagonal entries are largest, every hard prediction is persistence. This produces high accuracy when unchanged outcomes dominate, while change recall is zero. A stationary distribution pi satisfies pi P=pi and sum_i pi_i=1; its existence is general for a finite chain, but uniqueness and convergence require further conditions. It is not a short-term forecast conditioned on current physical pollution values. Phase 7 reuses the saved P, without making new claims about convergence.



## Existing Methods

The preserved project contains PCA/HoSVD compression, linear SVM, exact RBF SVM, RFF SVM, Markov persistence and fixed/adaptive/F3 probability fusion. E1 uses calibrated linear argmax; Phase 5 raw decisions are separately identified. Nested calibration is a separate prior experiment with insufficient row-level evidence here.



Fixed fusion is p_F=lambda p_S+(1-lambda)p_M. Adaptive fusion selects lambda_high when r_S>=tau and lambda_low otherwise, where r_S=1-p_S(i). The saved Phase 6 values are F1 lambda=0.9 with linear_hosvd_B; F2 tau=0.65, lambda_low=0.75, lambda_high=1 with linear_pca_B. F3 uses linear_best_overall, rho=0.75 and eta=0.

F3 separates change likelihood q_F3=rho[1-p_S(i)]+(1-rho)[1-p_M(i)] from destination. For j!=i, d(j)=[eta p_S(j)+(1-eta)p_M(j)]/[eta(1-p_S(i))+(1-eta)(1-p_M(i))], and d(i)=0. A zero denominator would require uniform mass over other states. The actual saved matrix has strictly positive off-diagonal row sums, so this contingency is not encountered. With eta=0, destination is entirely Markov-derived. That constraint can cause a correctly detected change to reach the wrong destination.



## Proposed Method

The locked choice is **E4**, classifier `temporal_C1.0_balanced`, alpha=0.75, with probability argmax decisions. It was selected using validation J=0.620206. The C grid was 0.1, 1 and 10; each was evaluated with no weighting and balanced weighting. E3/E4/E5 use 6/8/11 features respectively. E6 keeps E5's trained classifier and alpha and selects theta on validation. E7 fixes alpha=1.



Threshold features are d35=|m_t-35|, d75=|m_t-75| and dmin=min(d35,d75), followed by three one-hot current-state indicators. Though dmin is a deterministic nonlinear transformation of the other distances, it supplies a compact piecewise proximity basis to a linear logit. State indicators distinguish sides of thresholds. Temporal features are v3=(m_t-m_(t-3))/3 and v6=(m_t-m_(t-6))/6. These have units micrograms per cubic metre per hour. The correct window indices are 23,20,17.

For each of the other eleven stations v_s=(m_st-m_s(t-3))/3. Their mean, maximum and fraction strictly positive form the cross-station group. A contemporaneous association is not causal transport evidence. E3 uses six threshold/state features, E4 uses those plus two target trends, and E5/E6/E7 use all eleven. Fraction rising is dimensionless.

On rows with observed current state S_t, C_t=1[y_t!=S_t]. Let h'_t=(h_t-mu_h)/sigma_h, with both statistics fitted on training Scope A. Logistic regression gives q_P=1/(1+exp(-(w^T h'_t+b))). L2 regularization penalizes ||w||^2; C is inverse regularization strength. Class weights are either absent or balanced using training class frequencies. Balanced weights change the fitted effective class prior, so outputs should not automatically be regarded as calibrated population probabilities. No additional calibrator is fitted in this prespecified experiment; probability quality is measured and its deterioration disclosed.

For alpha in {0,.25,.5,.75,1}, q_new=alpha q_P+(1-alpha)q_F3. Set p(i)=1-q_new and p(j)=q_new d(j) for j!=i. Since 0<=q_new<=1 and the alternative destinations sum to one, sum_j p(j)=1-q_new+q_new=1 and p(j)>=0. Alpha=0 reproduces F3, and alpha=1 uses only the new change risk with the original destination model. Missing-state rows bypass these equations and use the original calibrated feature-only SVM.

Standard prediction is argmax p. E6 instead predicts change if q_new>=theta, then selects argmax d; otherwise it predicts i. Theta comes from {.2,.3,.4,.5,.6,.7} on validation, keeping E5's trained model and alpha. Probability arrays are not modified to force agreement with thresholded decisions. Argmax changes when q_new*max_j d(j)>1-q_new, so its effective transition threshold is 1/(1+max_j d(j)), generally above .5.



Algorithm: (1) verify hashes and align timestamps; (2) invert station-specific normalization; (3) construct threshold, temporal and other-station features; (4) train scalers and logistic models on observed-current training rows; (5) reconstruct F3 from saved models; (6) choose C, weighting and alpha by validation J; (7) choose E6 theta using validation only; (8) write locked configuration; (9) generate held-out probabilities/labels with missing-state SVM fallback; (10) independently recompute metrics and paired bootstrap intervals.

## Mathematical Analysis

The convex change mixture lies in [0,1]. The F3 alternative destination distribution sums to one, hence reconstructed probabilities sum to 1-q+q=1. Detection correctness is weaker than destination correctness, so A_change<=Recall_change. Accuracy decomposes into transition and stable components. See the complete mathematical summary for proofs and class-order conventions. No theorem guarantees improved prediction.

## Experimental Design

Chronological target windows: train 24,108; validation 4,239; test 5,720. The Phase 2 partitions are March 2013–December 2015, January–June 2016 and July 2016–February 2017. Windows are built inside splits; missing targets are excluded. Scope A validation/test counts are 4,197/5,685. No train/validation/test shuffling occurs.



Scope A includes observed current states; Scope B includes all next-hour targets. Transition metrics are always conditional on Scope A, even in a Scope B output row. Accuracy counts exact state matches divided by N. For each class k, precision=TP_k/(TP_k+FP_k), recall=TP_k/(TP_k+FN_k), and F1=2TP_k/(2TP_k+FP_k+FN_k); macro-F1 averages the three F1 values. Undefined zero-denominator event ratios are assigned zero.

Change TP means both true and predicted states differ from current state, even if the predicted destination is wrong. Change precision=TP/(TP+FP), recall=TP/(TP+FN), and F1=2TP/(2TP+FP+FN). Exact destination accuracy counts correctly predicted new states over actual transitions. Stable accuracy counts exact matches over unchanged events. Therefore exact destination accuracy cannot exceed change recall. With n_c transitions and n_s stable rows, Scope A accuracy=(n_c*A_change+n_s*A_stable)/(n_c+n_s).

High-entry truth is S_t!=2 and y_t=2; a predicted entry is S_t!=2 and prediction=2. Multiclass Brier loss is the mean sum over three squared probability errors. Log loss clips at 1e-12, renormalizes each row, and uses natural logarithms. Confidence ECE uses ten equal-width bins including 1 in the last bin, weighting absolute confidence-versus-correctness differences by bin frequency. ECE uses probability argmax correctness even for E6; it is not recalculated using thresholded labels.

Selection maximizes J=(macro-F1+change-F1)/2 on validation Scope A. Ties favor macro-F1, fewer features, smaller C, no weighting, then deterministic ID. The configuration is written before test predictions. Paired bootstrap samples calendar-day blocks with replacement using seed 42, reusing every sampled block for each model. Gaps from missing targets remain within their real calendar days. Percentile 2.5% and 97.5% quantiles define 95% intervals from 1,000 resamples. These intervals condition on fitted models and the selected specification; they do not account for training/selection uncertainty, dependence beyond one day, repeated historical use of this test set, or station-to-station generalization.


## Results

| model | accuracy | macro_f1 | change_precision | change_recall | change_f1 | exact_destination | stable_accuracy | high_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | 0.90748 | 0.88951 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 1.00000 | 0.00000 |
| E1 | 0.83694 | 0.80233 | 0.28339 | 0.46388 | 0.35184 | 0.41065 | 0.88040 | 0.30493 |
| E2 | 0.88461 | 0.86615 | 0.39601 | 0.26426 | 0.31699 | 0.15589 | 0.95891 | 0.03436 |
| E4 | 0.83518 | 0.80742 | 0.32096 | 0.55894 | 0.40777 | 0.40114 | 0.87943 | 0.30263 |

E4 detects 294 of 526 changes (55.89%) versus 139 (26.43%) for F3. False alarms increase from 212 to 622. Accuracy changes from 88.46% to 83.52%, and macro-F1 from 0.8661 to 0.8074. Correct destination accuracy increases from 15.59% to 40.11%, while stable-state accuracy declines from 95.89% to 87.94%. This is a transition-sensitivity trade-off, not an overall accuracy improvement.

Probability quality:

| model | brier | log_loss | ece |
| --- | --- | --- | --- |
| E0 | 0.16660 | 0.33916 | 0.01751 |
| E1 | 0.21965 | 0.37264 | 0.01503 |
| E2 | 0.18136 | 0.33289 | 0.06757 |
| E3 | 0.27612 | 0.48394 | 0.14432 |
| E4 | 0.27379 | 0.48032 | 0.14282 |
| E5 | 0.26570 | 0.46734 | 0.14118 |
| E6 | 0.26570 | 0.46734 | 0.14118 |
| E7 | 0.32126 | 0.54333 | 0.10578 |


Paired 95% differences against F3:

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


## Comparison

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

E5 improves the held-out transition trade-off relative to E4 but was not selected on validation. The full cross-phase evidence is in results/cross_phase_comparison.csv. No unmatched literature accuracy is compared numerically.

### Figure catalogue

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

## Limitations

Only one target station, one historical city dataset and one fixed chronological holdout are studied. The test set has already been used in previous project phases and the preserved legacy Phase 7 attempt; it is not a newly collected untouched external test set. Validation optimization over a finite grid can itself overfit. Intervals condition on trained models and the fixed selection and do not include training uncertainty. Calendar-day blocks may not capture multiday pollution episodes. No external station or prospective deployment test was performed.

The selected balanced logistic classifier overestimates change probabilities relative to observed frequency. No extra calibrator was fitted; Brier, log loss and ECE worsen relative to F3. This was not repaired by choosing a new configuration after inspecting test scores. A separately preregistered chronological out-of-fold calibration experiment is future work. The existing nested-calibration directory is a separate historical experiment whose lack of saved row-level predictions prevents independent recomputation here.

The original F3 destination distribution has eta=0, so destinations remain Markov-based. Detection can improve while destination errors remain. Physical histories include training-policy imputations, with flags exported. The fractions of rising stations reflect measured/imputed rates, not verified causal atmospheric transport. Hardware timing is a single warm local run for small new models, not an end-to-end deployment benchmark. Runtime and storage for unavailable historical measurements are N/A. The professor's authoritative original P2-P7 sheet is not present in the supplied local evidence, so compliance is mapped to the working descriptions only. No original 3,168-feature SVM run is fabricated.


## Conclusion

E4 detects 294 of 526 changes (55.89%) versus 139 (26.43%) for F3. False alarms increase from 212 to 622. Accuracy changes from 88.46% to 83.52%, and macro-F1 from 0.8661 to 0.8074. Correct destination accuracy increases from 15.59% to 40.11%, while stable-state accuracy declines from 95.89% to 87.94%. This is a transition-sensitivity trade-off, not an overall accuracy improvement. Highlight sensitivity and false-alarm costs together. Future work should independently evaluate chronological calibration and external stations before operational use.

### References

[1] A. Bekkar; B. Hssina; S. Douzi; K. Douzi. Air-pollution prediction in smart city, deep learning approach. Journal of Big Data, 2021. https://doi.org/10.1186/s40537-021-00548-1 (Primary publisher page opened).
[2] N. N. Zakaria; M. Othman; R. Sokkalingam; H. Daud; L. Abdullah; E. A. Kadir. Markov Chain Model Development for Forecasting Air Pollution Index of Miri, Sarawak. Sustainability, 2019. https://doi.org/10.3390/su11195190 (Original local PDF read; online publisher rate limited).
[3] C. Cortes; V. Vapnik. Support-vector networks. Machine Learning, 1995. https://doi.org/10.1007/BF00994018 (Primary publisher metadata and abstract opened).
[4] A. Rahimi; B. Recht. Random Features for Large-Scale Kernel Machines. Advances in Neural Information Processing Systems, 2007. https://proceedings.neurips.cc/paper_files/paper/2007/file/013a006f03dbc5392effeb8f18fda755-Paper.pdf (Primary proceedings paper opened).
[5] L. De Lathauwer; B. De Moor; J. Vandewalle. A Multilinear Singular Value Decomposition. SIAM Journal on Matrix Analysis and Applications, 2000. https://doi.org/10.1137/S0895479896305696 (Primary publisher metadata and abstract opened).
[6] T. G. Kolda; B. W. Bader. Tensor Decompositions and Applications. SIAM Review, 2009. https://doi.org/10.1137/07070111X (Primary publisher metadata and abstract opened).
[7] C. Guo; G. Pleiss; Y. Sun; K. Q. Weinberger. On Calibration of Modern Neural Networks. ICML / PMLR 70, 2017. https://proceedings.mlr.press/v70/guo17a.html (Primary proceedings metadata and abstract opened).
[8] A. Niculescu-Mizil; R. Caruana. Predicting Good Probabilities With Supervised Learning. ICML, 2005. https://doi.org/10.1145/1102351.1102430 (Primary conference PDF opened).
[9] I. T. Jolliffe; J. Cadima. Principal component analysis: a review and recent developments. Philosophical Transactions of the Royal Society A, 2016. https://doi.org/10.1098/rsta.2015.0202 (Publisher search metadata verified; full-page access blocked).
[10] S. Zhang; B. Guo; A. Dong; J. He; Z. Xu; S. X. Chen. Cautionary tales on air-quality improvement in Beijing. Proceedings of the Royal Society A, 2017. https://doi.org/10.1098/rspa.2017.0457 (Primary article archived at PMC opened).
[11] R.-E. Fan; K.-W. Chang; C.-J. Hsieh; X.-R. Wang; C.-J. Lin. LIBLINEAR: A Library for Large Linear Classification. Journal of Machine Learning Research, 2008. https://jmlr.org/papers/v9/fan08a.html (Primary journal page opened).
[12] H. R. Kunsch. The Jackknife and the Bootstrap for General Stationary Observations. The Annals of Statistics, 1989. https://doi.org/10.1214/aos/1176347265 (WARNING: metadata corroborated; primary publisher blocked).
[13] C.-C. Chang; C.-J. Lin. LIBSVM: A Library for Support Vector Machines. ACM Transactions on Intelligent Systems and Technology, 2011. https://doi.org/10.1145/1961189.1961199 (Publisher and author search metadata verified; PDF timed out).
[14] T. Saito; M. Rehmsmeier. The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. PLOS ONE, 2015. https://doi.org/10.1371/journal.pone.0118432 (Primary publisher article opened).
[15] G. C. Cawley; N. L. C. Talbot. On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation. Journal of Machine Learning Research, 2010. https://jmlr.org/papers/v11/cawley10a.html (Primary journal page opened).
