# Mathematical specification and evidence

## Data, notation and states

Let t denote the final observed hour, s a monitoring station and f a numerical channel. A history tensor X_t has shape 24 by 12 by 11. Its rows run from t-23 through t; the label y_t is the state at t+1. The target station is Aotizhongxin, station index 0; PM2.5 is channel 0. A state is 0 for m<35, 1 for 35<=m<75, and 2 for m>=75, with m in micrograms per cubic metre. These instantaneous project categories are not official AQI categories. Missing observed current measurements retain state -1.

Phase 2 forward-fills no more than six hours within each chronological partition, then uses training medians. It standardizes each station-channel independently. For training mean mu_sf and standard deviation sigma_sf, z_tsf=(x_tsf-mu_sf)/sigma_sf. The inverse is x_tsf=z_tsf sigma_sf+mu_sf. Float32 storage makes inverse values approximate; the experiment checks observed historical values within 0.001 micrograms per cubic metre. Imputed histories cannot establish an observed current state. New scalers are fitted on training rows with observed current states only.

## SVD and PCA

Flatten a centered history into a row in R^3168, forming the training matrix A. Its singular value decomposition is A=U Sigma V^T, with orthonormal columns U,V and nonnegative singular values in Sigma. For retained dimension k, the feature row is a V_k and reconstruction is a V_k V_k^T. Add back the training mean to reconstruct uncentered values. The first k right singular vectors maximize captured training variance. For the truncated matrix SVD, squared Frobenius reconstruction error equals the sum of squared discarded singular values. This optimality concerns unconstrained rank-k matrix approximation and does not imply best predictive accuracy.

The implementation in AP/phase4_output/phase4_pca_hosvd.py fits training means and accumulated Gram matrices; eig_desc obtains principal axes, evaluate_split measures error, and storage_numbers separates basis and feature storage. A flattened input needs 3,168 numbers per sample; a code needs k. The logical code-only compression ratio is 3168/k. Total compression must include the mean and 3168*k basis entries, their precision, and the number of samples. On-disk compressed NPZ sizes are a different quantity.

Relative reconstruction error is ||A-A_hat||_F / ||A||_F. Mean squared error divides squared error by the number of scalar entries. Retained energy is ||A_hat||_F^2 / ||A||_F^2 for orthogonal projection. Compare methods on the same centered data and split. A reconstruction-versus-dimension curve is not a classification-versus-dimension curve; both are supplied separately.

## HoSVD and tensor compression

For a third-order tensor T, mode-n unfolding T_(n) arranges mode-n fibers as columns of a matrix. Orthonormal mode matrices U_1, U_2, U_3 span retained time, station and channel subspaces. With ranks r_1,r_2,r_3, the core G=T times_1 U_1^T times_2 U_2^T times_3 U_3^T has shape r_1 by r_2 by r_3. The symbol times_n denotes multiplication along tensor mode n; other modes retain their dimensions. Reconstruction is G times_1 U_1 times_2 U_2 times_3 U_3. Flattening the core gives k=r_1*r_2*r_3 SVM features.

The shared mode matrices in this project are fitted from aggregated training mode Gram matrices, not separately fitted on test samples. Their orthogonal projections preserve the tensor axes. The usual squared projection-error upper bound is the sum, over modes, of the discarded mode singular-value energies. For the stacked training collection this follows by sequential orthogonal projections and a telescoping decomposition: each residual term is bounded by the corresponding one-mode projection residual. A truncated HoSVD is generally not the optimal multilinear-rank approximation. Do not transfer the matrix SVD optimality theorem to arbitrary tensor rank constraints.

The stored mode basis requires 24*r_1+12*r_2+11*r_3 numbers, plus the mean and per-sample core. The code-only ratio is 3168/(r_1*r_2*r_3). A mode rank cannot exceed its original mode dimension. AP/phase4_output/hosvd_truncation_bound.csv and pca_hosvd_comparison.csv preserve historical rank/error evidence. No earlier compression model is retrained by Phase 7.

## SVM and kernels

For binary labels a_i in {-1,+1}, a linear decision function is f(x)=w^T x+b. Geometric margin is |f(x)|/||w||. The conventional soft-margin formulation minimizes (1/2)||w||^2+C sum_i xi_i subject to a_i f(x_i)>=1-xi_i and xi_i>=0. C trades margin regularization against training violations. LinearSVC uses its configured loss and multiclass strategy; its common squared-hinge objective replaces linear slack penalty by a squared penalty. The report must distinguish this implementation detail from the conventional introductory primal.

The RBF kernel K(x,x')=exp(-gamma ||x-x'||^2) measures similarity without explicitly constructing an infinite-dimensional feature vector. gamma controls locality. An exact SVC trains kernel classifiers, with internal one-versus-one multiclass handling; requesting an OvR-shaped decision output does not retrain it as one-versus-rest. LinearSVC uses one-versus-rest in this saved project model. Class order is verified as [0,1,2]. A hard margin decision and a calibrated probability argmax can differ. E1 uses the calibrated argmax used by Phase 6; historical Phase 5 hard decisions are reported separately.

## Random Fourier features

For the above RBF kernel, draw omega_j from N(0,2 gamma I) and b_j uniformly on [0,2*pi]. Define z_j(x)=sqrt(2/D) cos(omega_j^T x+b_j), j=1,...,D. The expectation of z(x)^T z(x') equals the RBF kernel. Finite D introduces approximation error; larger D usually improves approximation in expectation, not necessarily each observed classification score. Fit a linear classifier on z(x). Prediction work scales with input dimension times D plus linear classifier work; exact kernel prediction depends on support-vector count. Random seeds and mapping parameters must be preserved. The historical RFF models and revised experiments remain separate; no new random map is substituted for a saved one.

## Markov chain

Let N_ij count consecutive valid observed training-state pairs i to j separated by exactly one hour. P_ij=N_ij/sum_j N_ij. Every row sums to one and P_ij>=0. For known current state i, p_M(j)=P_ij. The hard prediction maximizes that row. When diagonal entries are largest, every hard prediction is persistence. This produces high accuracy when unchanged outcomes dominate, while change recall is zero. A stationary distribution pi satisfies pi P=pi and sum_i pi_i=1; its existence is general for a finite chain, but uniqueness and convergence require further conditions. It is not a short-term forecast conditioned on current physical pollution values. Phase 7 reuses the saved P, without making new claims about convergence.

## Historical fusion

Fixed fusion is p_F=lambda p_S+(1-lambda)p_M. Adaptive fusion selects lambda_high when r_S>=tau and lambda_low otherwise, where r_S=1-p_S(i). The saved Phase 6 values are F1 lambda=0.9 with linear_hosvd_B; F2 tau=0.65, lambda_low=0.75, lambda_high=1 with linear_pca_B. F3 uses linear_best_overall, rho=0.75 and eta=0.

F3 separates change likelihood q_F3=rho[1-p_S(i)]+(1-rho)[1-p_M(i)] from destination. For j!=i, d(j)=[eta p_S(j)+(1-eta)p_M(j)]/[eta(1-p_S(i))+(1-eta)(1-p_M(i))], and d(i)=0. A zero denominator would require uniform mass over other states. The actual saved matrix has strictly positive off-diagonal row sums, so this contingency is not encountered. With eta=0, destination is entirely Markov-derived. That constraint can cause a correctly detected change to reach the wrong destination.

## Phase 7 features and logistic change model

Threshold features are d35=|m_t-35|, d75=|m_t-75| and dmin=min(d35,d75), followed by three one-hot current-state indicators. Though dmin is a deterministic nonlinear transformation of the other distances, it supplies a compact piecewise proximity basis to a linear logit. State indicators distinguish sides of thresholds. Temporal features are v3=(m_t-m_(t-3))/3 and v6=(m_t-m_(t-6))/6. These have units micrograms per cubic metre per hour. The correct window indices are 23,20,17.

For each of the other eleven stations v_s=(m_st-m_s(t-3))/3. Their mean, maximum and fraction strictly positive form the cross-station group. A contemporaneous association is not causal transport evidence. E3 uses six threshold/state features, E4 uses those plus two target trends, and E5/E6/E7 use all eleven. Fraction rising is dimensionless.

On rows with observed current state S_t, C_t=1[y_t!=S_t]. Let h'_t=(h_t-mu_h)/sigma_h, with both statistics fitted on training Scope A. Logistic regression gives q_P=1/(1+exp(-(w^T h'_t+b))). L2 regularization penalizes ||w||^2; C is inverse regularization strength. Class weights are either absent or balanced using training class frequencies. Balanced weights change the fitted effective class prior, so outputs should not automatically be regarded as calibrated population probabilities. No additional calibrator is fitted in this prespecified experiment; probability quality is measured and its deterioration disclosed.

For alpha in {0,.25,.5,.75,1}, q_new=alpha q_P+(1-alpha)q_F3. Set p(i)=1-q_new and p(j)=q_new d(j) for j!=i. Since 0<=q_new<=1 and the alternative destinations sum to one, sum_j p(j)=1-q_new+q_new=1 and p(j)>=0. Alpha=0 reproduces F3, and alpha=1 uses only the new change risk with the original destination model. Missing-state rows bypass these equations and use the original calibrated feature-only SVM.

Standard prediction is argmax p. E6 instead predicts change if q_new>=theta, then selects argmax d; otherwise it predicts i. Theta comes from {.2,.3,.4,.5,.6,.7} on validation, keeping E5's trained model and alpha. Probability arrays are not modified to force agreement with thresholded decisions. Argmax changes when q_new*max_j d(j)>1-q_new, so its effective transition threshold is 1/(1+max_j d(j)), generally above .5.

## Evaluation and uncertainty

Scope A includes observed current states; Scope B includes all next-hour targets. Transition metrics are always conditional on Scope A, even in a Scope B output row. Accuracy counts exact state matches divided by N. For each class k, precision=TP_k/(TP_k+FP_k), recall=TP_k/(TP_k+FN_k), and F1=2TP_k/(2TP_k+FP_k+FN_k); macro-F1 averages the three F1 values. Undefined zero-denominator event ratios are assigned zero.

Change TP means both true and predicted states differ from current state, even if the predicted destination is wrong. Change precision=TP/(TP+FP), recall=TP/(TP+FN), and F1=2TP/(2TP+FP+FN). Exact destination accuracy counts correctly predicted new states over actual transitions. Stable accuracy counts exact matches over unchanged events. Therefore exact destination accuracy cannot exceed change recall. With n_c transitions and n_s stable rows, Scope A accuracy=(n_c*A_change+n_s*A_stable)/(n_c+n_s).

High-entry truth is S_t!=2 and y_t=2; a predicted entry is S_t!=2 and prediction=2. Multiclass Brier loss is the mean sum over three squared probability errors. Log loss clips at 1e-12, renormalizes each row, and uses natural logarithms. Confidence ECE uses ten equal-width bins including 1 in the last bin, weighting absolute confidence-versus-correctness differences by bin frequency. ECE uses probability argmax correctness even for E6; it is not recalculated using thresholded labels.

Selection maximizes J=(macro-F1+change-F1)/2 on validation Scope A. Ties favor macro-F1, fewer features, smaller C, no weighting, then deterministic ID. The configuration is written before test predictions. Paired bootstrap samples calendar-day blocks with replacement using seed 42, reusing every sampled block for each model. Gaps from missing targets remain within their real calendar days. Percentile 2.5% and 97.5% quantiles define 95% intervals from 1,000 resamples. These intervals condition on fitted models and the selected specification; they do not account for training/selection uncertainty, dependence beyond one day, repeated historical use of this test set, or station-to-station generalization.
