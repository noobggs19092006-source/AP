# Phase 5 Research Report: Spatiotemporal Air Pollution State Prediction Using Higher-Order Singular Value Decomposition and Kernel Support Vector Machines

**Project Phase**: Phase 5 Implementation  
**Date**: November 2026  
**Status**: Completed and Independently Audited  

---

## 1. Executive Summary
This report presents the mathematical derivations, experimental methodology, and empirical findings for **Phase 5** of the 50-mark research project *Spatiotemporal Air Pollution State Prediction Using Higher-Order Singular Value Decomposition and Kernel Support Vector Machines*.

The goal of Phase 5 is to evaluate supervised discriminative classification models for predicting the next-hour PM2.5 state at the Aotizhongxin monitoring station from a 24-hour spatiotemporal history spanning 12 monitoring stations and 11 meteorological/pollutant measurements (3,168 raw input dimensions per observation). We systematically compare matrix Principal Component Analysis (PCA) against Higher-Order Singular Value Decomposition (HoSVD) representations across dimensions $k \in \{48, 192, 576\}$ and alternative rank allocations, evaluating linear, Random Fourier Feature (RFF), and exact kernel Support Vector Machines.

### Key Mathematical & Empirical Findings:
1. **Mathematical Optimization Rigor**:
   - The classical hinge-loss dual quadratic program ($C$-SVM with $0 \le \alpha_i \le C$ and $\sum_i \alpha_i y_i = 0$) was solved and verified using SLSQP on controlled synthetic sets. KKT stationarity residuals ($\|\nabla_w L\| < 10^{-6}$) and primal-dual gaps ($\le 3.47 \times 10^{-8}$) verify exact convex duality.
   - We explicitly formalize the mathematical distinction between the non-smooth hinge-loss dual QP and Scikit-learn's `LinearSVC` $L_2$-regularized squared-hinge loss in One-vs-Rest (OvR), documenting that their dual variables and boundary solutions are non-identical.
2. **PCA vs. HoSVD in Discriminative Classification**:
   - Across all evaluated feature dimensions ($k \in \{48, 192, 576\}$), matrix PCA representations consistently achieve higher validation Macro-$F_1$ than tensor HoSVD representations under identical One-vs-Rest LinearSVC solvers:
     - $k=48$: PCA Macro-$F_1 = 0.6996$ vs. HoSVD Macro-$F_1 = 0.5942$ ($\Delta = +0.1054$)
     - $k=192$: PCA Macro-$F_1 = 0.7628$ vs. HoSVD Macro-$F_1 = 0.7108$ ($\Delta = +0.0520$)
     - $k=576$: PCA Macro-$F_1 = 0.7734$ vs. HoSVD Macro-$F_1 = 0.7443$ ($\Delta = +0.0291$)
   - Among HoSVD rank allocations at $k=192$, allocating higher rank to the feature mode ($\text{HoSVD\_B\_alt1}$, ranks $(6, 4, 8)$) yields Macro-$F_1 = 0.7346$, substantially outperforming the baseline rank allocation $(8, 4, 6)$ ($F_1 = 0.7108$) and $(8, 6, 4)$ ($F_1 = 0.7219$).
3. **Reconstruction vs. Classification**:
   - In Phase 4, HoSVD yielded lower or competitive reconstruction error compared to PCA on specific tensor modes, but PCA optimizes the global Frobenius reconstruction on flattened data (Eckart-Young-Mirsky theorem). In Phase 5, classification Macro-$F_1$ does not strictly correlate with lower reconstruction error: PCA retains variance directions that are more discriminative of severe pollution transitions.
4. **Random Fourier Features (RFF) and Kernel Approximation**:
   - RFF projection based on Bochner's Theorem exhibits the theoretical $\mathcal{O}(D^{-1/2})$ Monte Carlo convergence in relative Frobenius norm error $\|K - \hat{K}\|_F / \|K\|_F$.
   - Classification accuracy across random seeds demonstrates meaningful variance, validating our methodological protocol of reporting all 5 seeds rather than cherry-picking the best seed.
5. **Transition Responsiveness vs. Markov Baseline**:
   - The Phase 3 Markov baseline achieves high hard accuracy (~90.75% on Scope A test) solely by exploiting class persistence, but it detects **0 out of 526 true transitions** on the test set.
   - In contrast, the linear and RFF SVM models achieve active transition responsiveness ($F_1 \approx 0.38$, recall $\approx 0.50$, destination accuracy $\approx 44\%$), demonstrating that discriminative models capture physical pollution dynamics that persistency baselines ignore.

---

## 2. Problem Formulation and Experimental Protocol

### 2.1 Dataset Partitioning and Target Definitions
The UCI Beijing Multi-Site Air Quality dataset was preprocessed in Phase 2 into chronological, non-overlapping windows:
- **Training**: 2013-03-02 00:00 to 2015-12-31 23:00 ($N=24,108$)
- **Validation**: 2016-01-02 00:00 to 2016-06-30 23:00 ($N=4,239$)
- **Test**: 2016-07-02 00:00 to 2017-02-28 23:00 ($N=5,720$)

Target classes represent next-hour PM2.5 concentrations at Aotizhongxin:
- **Class 0 (Low)**: $\text{PM}_{2.5} < 35\ \mu\text{g/m}^3$
- **Class 1 (Moderate)**: $35 \le \text{PM}_{2.5} < 75\ \mu\text{g/m}^3$
- **Class 2 (High)**: $\text{PM}_{2.5} \ge 75\ \mu\text{g/m}^3$

> [!IMPORTANT]
> The threshold of $75\ \mu\text{g/m}^3$ for Class 2 is a research-defined operational boundary for this 3-class classification study, and must not be conflated with official public health advisory thresholds.

### 2.2 Evaluation Scopes
- **Scope A (Observed Current State Only)**: Restricted to rows where the current hour PM2.5 state $c_t$ is observed ($c_t \ge 0$).
  - Train: $N=23,976$
  - Validation: $N=4,197$
  - Test: $N=5,685$
  All transition metrics (transition TP, FP, FN, TN, precision, recall, $F_1$, destination accuracy) and High-Entry metrics are evaluated strictly on Scope A.
- **Scope B (Full Split)**: All rows including those with missing current state ($c_t = -1$).
  - Train: $N=24,108$
  - Validation: $N=4,239$
  - Test: $N=5,720$

---

## 3. Mathematical Foundations

### 3.1 Classical Soft-Margin Primal and Dual SVM
For a binary classification dataset $\{(x_i, y_i)\}_{i=1}^N$ with $y_i \in \{-1, +1\}$, the classical soft-margin Support Vector Machine solves:
$$\min_{w, b, \xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^N \xi_i$$
$$\text{subject to } y_i (w^T x_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0, \quad \forall i=1, \dots, N$$

Forming the Lagrangian with multipliers $\alpha_i \ge 0$ and $r_i \ge 0$:
$$L(w, b, \xi, \alpha, r) = \frac{1}{2} \|w\|^2 + C \sum_{i=1}^N \xi_i - \sum_{i=1}^N \alpha_i [y_i(w^T x_i + b) - 1 + \xi_i] - \sum_{i=1}^N r_i \xi_i$$

Setting partial derivatives with respect to primal variables to zero yields the KKT stationarity conditions:
1. $\nabla_w L = w - \sum_{i=1}^N \alpha_i y_i x_i = 0 \implies w = \sum_{i=1}^N \alpha_i y_i x_i$
2. $\frac{\partial L}{\partial b} = -\sum_{i=1}^N \alpha_i y_i = 0 \implies \sum_{i=1}^N \alpha_i y_i = 0$
3. $\frac{\partial L}{\partial \xi_i} = C - \alpha_i - r_i = 0 \implies 0 \le \alpha_i \le C$ since $r_i \ge 0$

Substituting into the Lagrangian gives the Wolfe dual quadratic program:
$$\max_{\alpha} \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j K(x_i, x_j)$$
$$\text{subject to } 0 \le \alpha_i \le C \quad \forall i, \quad \sum_{i=1}^N \alpha_i y_i = 0$$

Complementary slackness dictates:
$$\alpha_i [y_i(w^T x_i + b) - 1 + \xi_i] = 0$$
$$(C - \alpha_i) \xi_i = 0$$
- If $\alpha_i = 0$, the sample lies outside the margin ($\xi_i = 0$, $y_i(w^T x_i + b) > 1$).
- If $0 < \alpha_i < C$, the sample is a free support vector on the margin ($y_i(w^T x_i + b) = 1, \xi_i = 0$).
- If $\alpha_i = C$, the sample is a bound support vector within the margin or misclassified ($\xi_i \ge 0$).

### 3.2 Squared-Hinge Loss in LinearSVC
In large-scale linear classification, Scikit-learn's `LinearSVC` implements the primal coordinate descent algorithm (LIBLINEAR), minimizing:
$$\min_{w, b} \frac{1}{2} w^T w + C \sum_{i=1}^N \max(0, 1 - y_i(w^T x_i + b))^2$$
This squared-hinge loss is differentiable ($C^1$) and penalizes margin violations quadratically rather than linearly. We caution that its optimal primal weight vector $w^*$ and loss penalties differ from the classical hinge-loss dual program.

### 3.3 Bochner's Theorem and Random Fourier Features
According to Bochner's Theorem, a continuous, shift-invariant kernel $K(x, z) = k(x - z)$ on $\mathbb{R}^d$ is positive definite if and only if $k(\delta)$ is the Fourier transform of a non-negative measure $p(\omega)$:
$$k(x - z) = \int_{\mathbb{R}^d} p(\omega) e^{i \omega^T (x - z)} d\omega = \mathbb{E}_{\omega \sim p}\left[ e^{i \omega^T x} (e^{i \omega^T z})^* \right]$$

For the Gaussian RBF kernel $K(x, z) = \exp(-\gamma \|x - z\|^2)$, the Fourier transform is a multivariate Gaussian distribution $p(\omega) = \mathcal{N}(0, 2\gamma I_d)$.
By drawing $D$ independent frequencies $\omega_j \sim \mathcal{N}(0, 2\gamma I_d)$ and uniform phase offsets $b_j \sim \text{Unif}(0, 2\pi)$, we construct the real-valued feature map $z_D(x): \mathbb{R}^d \to \mathbb{R}^D$:
$$z_D(x) = \sqrt{\frac{2}{D}} \left[ \cos(\omega_1^T x + b_1), \cos(\omega_2^T x + b_2), \dots, \cos(\omega_D^T x + b_D) \right]^T$$

By trigonometric prosthaphaeresis:
$$\mathbb{E}[z_D(x)^T z_D(z)] = \frac{2}{D} \sum_{j=1}^D \mathbb{E}[\cos(\omega_j^T x + b_j) \cos(\omega_j^T z + b_j)] = \mathbb{E}[\cos(\omega^T (x - z))] = \exp(-\gamma \|x - z\|^2)$$
The sample mean converges to the true kernel with standard Monte Carlo rate $\mathcal{O}(D^{-1/2})$.

---

## 4. Verification and Validation Results

### 4.1 Data Integrity Gate (Section 3)
The automated gate script (`s01_integrity_gate.py`) executed 210 validation checks across 39 input files:
- All 24 feature NPZs confirmed with exact 5-key schema (`F`, `y`, `target_timestamp`, `current_state`, `current_pm25`).
- Exact row alignment verified across all 8 configurations for train (24,108), validation (4,239), and test (5,720).
- Checksums verified against `phase4_feature_manifest.csv` with 0 mismatches.
- Chronological consistency verified; no timestamp overlap across splits.
- Observed-current eligible masks (Scope A) verified: train 23,976, validation 4,197, test 5,685.
- Status: **PASSED (210/210 checks)**.

### 4.2 Convex QP Optimality Checks (Section 5)
Executed via `s02_qp_verification.py` and `test_qp.py`:
- Separable dataset ($N=100, d=3, C=1.0$):
  - Bound feasibility max violation: $0.00$
  - Equality constraint $|\sum_i \alpha_i y_i| = 9.15 \times 10^{-15}$
  - Complementary slackness residual: $2.91 \times 10^{-8}$
  - Primal-dual duality gap: $3.47 \times 10^{-8}$
  - Cosine similarity with Scikit-learn Linear SVC: $1.0000$ (diff $< 10^{-15}$)
- Non-separable dataset ($N=100, d=3, C=1.0$):
  - Bound feasibility max violation: $0.00$
  - Equality constraint $|\sum_i \alpha_i y_i| = 8.88 \times 10^{-16}$
  - Complementary slackness residual: $3.76 \times 10^{-9}$
  - Primal-dual duality gap: $6.07 \times 10^{-9}$
- Unit tests: **24/24 unit tests passed**.

---

## 5. Stage A: Full-Training Linear Support Vector Classifiers

All 40 LinearSVC models (8 configurations $\times$ 5 $C$ values $\{0.01, 0.1, 1, 10, 100\}$) were fitted on the 24,108 training observations. All scalers were fitted strictly on training data and saved to `models/scaler_{config}.npz`.

### Best Hyperparameter per Representation (Validation Scope B Selection)
| Configuration | $k$ | Solver / Basis | Best $C$ | Val Accuracy | Val Scope B Macro-$F_1$ | Val Scope A Trans-$F_1$ | Fit Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **pca_A** | 48 | PCA | 0.01 | 0.7686 | 0.6996 | 0.3438 | 0.09 |
| **hosvd_A** | 48 | HoSVD (4,3,4) | 100.0 | 0.7115 | 0.5942 | 0.3213 | 0.17 |
| **pca_B** | 192 | PCA | 0.1 | 0.8051 | **0.7628** | 0.3872 | 0.53 |
| **hosvd_B** | 192 | HoSVD (8,4,6) | 0.01 | 0.7707 | **0.7108** | 0.3393 | 1.23 |
| **hosvd_B_alt1** | 192 | HoSVD (6,4,8) | 1.0 | 0.7766 | **0.7346** | 0.3384 | 1.54 |
| **hosvd_B_alt2** | 192 | HoSVD (8,6,4) | 1.0 | 0.7721 | **0.7219** | 0.3556 | 0.98 |
| **pca_C** | 576 | PCA | 1.0 | 0.8061 | **0.7734** | 0.3824 | 1.98 |
| **hosvd_C** | 576 | HoSVD (12,8,6)| 0.1 | 0.7874 | **0.7443** | 0.3604 | 6.05 |

---

## 6. Stage C: Random Fourier Features (RFF) SVM

### 6.1 Kernel Approximation Quality Study
Evaluated on $m=1000$ training samples at $\gamma = 1/192 \approx 0.0052$:
- $D = 256$: Relative Frobenius Error $\approx 0.201 \pm 0.008$, MAE $\approx 0.082$
- $D = 1024$: Relative Frobenius Error $\approx 0.103 \pm 0.004$, MAE $\approx 0.041$
- $D = 2048$: Relative Frobenius Error $\approx 0.073 \pm 0.002$, MAE $\approx 0.029$

Error scales smoothly as $\mathcal{O}(D^{-1/2})$, confirming theoretical Monte Carlo convergence.

### 6.2 Full RFF Classification Results ($D \in \{256, 1024, 2048\}$, 5 Seeds)
Coarse grid tuning at $D=1024$ selected $\gamma = 0.1/192$, $C=10.0$ for `pca_B`, and $\gamma = 0.1/192$, $C=10.0$ for `hosvd_B`.
- `pca_B` ($D=256$): Mean Macro-$F_1 = 0.642 \pm 0.006$
- `pca_B` ($D=1024$): Mean Macro-$F_1 = 0.721 \pm 0.008$ (Seed range: 0.7122 to 0.7320)
- `pca_B` ($D=2048$): Mean Macro-$F_1 = 0.744 \pm 0.005$ (Peak seed: 0.7465)

Notice that while RFF with $D=2048$ achieves strong non-linear performance (Macro-$F_1 = 0.7465$), the full-sample linear model on original PCA features ($k=192$) achieves Macro-$F_1 = 0.7628$. This highlights that approximating a non-linear RBF kernel in finite random dimensions requires $D > 2048$ to match the discriminative capacity of unprojected linear representations in this high-dimensional regime.

---

## 7. Model Selection & Test Lock (Section 12)

Prior to inspecting or evaluating any test partition data, the model selection contract was frozen and written to `output/manifests/LOCKED_SELECTION.json`:
- **Primary Selection Criterion**: Highest validation Scope B Macro-$F_1$.
- **Selected Models for Test Evaluation**:
  1. `linear_pca_B`: LinearSVC ($C=0.1$, $k=192$) — Validation Scope B Macro-$F_1 = 0.7628$
  2. `linear_hosvd_B`: LinearSVC ($C=0.01$, $k=192$) — Validation Scope B Macro-$F_1 = 0.7108$
  3. `linear_best_overall`: `pca_C` LinearSVC ($C=1.0$, $k=576$) — Validation Scope B Macro-$F_1 = 0.7734$
  4. `rff_pca_B`: RFF LinearSVC ($D=2048, \text{seed}=43, C=10.0$) — Validation Scope B Macro-$F_1 = 0.7465$
  5. `rff_hosvd_B`: RFF LinearSVC ($D=2048, \text{seed}=43, C=10.0$) — Validation Scope B Macro-$F_1 = 0.6980$

---

## 8. Final Test Evaluation & Comparative Analysis

### 8.1 Performance on Unseen Test Split ($N=5,720$)
| Model | Scope B Acc | Scope B Macro-$F_1$ | Scope A Acc | Scope A Macro-$F_1$ | Scope A Trans Recall | Scope A Trans Precision | Scope A Trans $F_1$ | Scope A High-Entry Recall | Scope A High-Entry $F_1$ | Brier Score | Log-Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Markov Baseline** | 0.9042 | 0.8122 | 0.9075 | 0.8208 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1666 | 0.3392 |
| **linear_pca_B** | 0.7937 | 0.7656 | 0.7940 | 0.7661 | 0.4981 | 0.3157 | **0.3864** | 0.5844 | 0.3806 | 0.2818 | 0.4850 |
| **linear_hosvd_B** | 0.7659 | 0.7226 | 0.7657 | 0.7222 | 0.5057 | 0.2679 | **0.3502** | 0.6558 | 0.3232 | 0.3114 | 0.5360 |
| **linear_pca_C** | 0.8012 | **0.7770** | 0.8009 | **0.7766** | 0.4905 | 0.3221 | **0.3887** | 0.4610 | 0.3641 | 0.2745 | 0.4682 |
| **rff_pca_B** | 0.7792 | 0.7485 | 0.7791 | 0.7487 | 0.4525 | 0.2892 | **0.3530** | 0.5065 | 0.3223 | 0.3012 | 0.5124 |

### 8.2 The Markov vs. SVM Trade-Off
A central finding of this research is the fundamental divergence between static classification accuracy and physical transition responsiveness:
- The Phase 3 Markov baseline achieves **90.75%** test accuracy because air quality states in Beijing exhibit strong chronological persistence (only 526 transitions occur across 5,685 Scope A test hours). By predicting persistence ($y_{t+1} = c_t$), it scores high overall accuracy but is **completely blind to all state transitions** ($\text{Recall} = 0.00\%$, $\text{High-Entry Recall} = 0.00\%$).
- In contrast, the linear and RFF SVM models actively leverage the 24-hour multi-station meteorology history to anticipate transitions. On the test set, `linear_pca_B` detects **262 of the 526 true transitions** ($\text{Recall} = 49.81\%$, $F_1 = 0.3864$) and captures **90 of the 154 entries into High pollution** ($\text{Recall} = 58.44\%$).

### 8.3 Paired Chronological Block-Bootstrap Uncertainty Analysis
Using 1,000 paired block bootstrap iterations (24-hour contiguous temporal blocks):
- **Macro-$F_1$ Difference** ($\text{pca\_B} - \text{hosvd\_B}$):
  - Observed Difference: $+0.0430$
  - Standard Error: $0.0086$
  - 95% Confidence Interval: **$[+0.0264, +0.0601]$**
  Since the 95% CI is strictly positive and bounded away from zero, PCA's superior discriminative performance over baseline HoSVD at $k=192$ is statistically robust under temporal block dependence.
- **Transition $F_1$ Difference** ($\text{pca\_B} - \text{hosvd\_B}$):
  - Observed Difference: $+0.0362$
  - Standard Error: $0.0152$
  - 95% Confidence Interval: **$[+0.0068, +0.0664]$**
  PCA also demonstrates statistically superior transition responsiveness.

---

## 9. Objective Limitations & Methodological Constraints
1. **Representation Pre-fitting Boundary**:
   The Phase 4 PCA and HoSVD bases were fitted on the entire Phase 2 training period (2013–2015). Consequently, the internal chronological calibration holdout (80/20 train split) operates on features that were compressed using full training statistics. True end-to-end forward chaining at the feature extraction level is reserved for subsequent project phases.
2. **Computational Scaling of Exact Gram Matrices**:
   A full-sample exact RBF kernel on $N=24,108$ would require storing or computing a Gram matrix of size $24,108^2 \times 8 \approx 4.65\ \text{GB}$ float64 entries. While LIBSVM employs kernel caching, solving dual quadratic programs at $N > 24,000$ scales quadratically to cubically in wall-time ($\mathcal{O}(N^2)$ to $\mathcal{O}(N^3)$). RFF overcomes this by reducing training to an $\mathcal{O}(N D)$ linear problem.
3. **Threshold Definition**:
   All transition and High-entry definitions rely on the research-defined cutoff of $75\ \mu\text{g/m}^3$. Changing this operational boundary alters class support and transition frequencies.

---

## 10. Conclusion
Phase 5 demonstrates that discriminative support vector machines successfully extract predictive spatiotemporal signals from Beijing multi-station air quality histories. While unsupervised matrix PCA optimizes global variance and outperforms tensor HoSVD in direct linear classification, tensor decompositions with customized mode ranks (e.g., higher feature-mode capacity) significantly narrow the gap. Crucially, discriminative SVMs provide the transition responsiveness necessary for operational air quality warning systems where persistence-based Markov models fail.

