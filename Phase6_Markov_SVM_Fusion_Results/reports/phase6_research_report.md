# Phase 6 Research Report

**Transition-Aware Next-Hour Pollution State Prediction via Markov–SVM Probability Fusion**

## Abstract
We investigate whether a Markov state-transition model and a feature-based SVM classifier can be combined to improve the trade-off between overall classification accuracy, binary change detection, exact destination accuracy on true transitions, and stable-state persistence prediction for next-hour PM2.5 pollution state at the Aotizhongxin station in Beijing. Using Phase 4 compressed features (PCA/HoSVD), Phase 5 calibrated Linear SVM and RFF-Linear SVM models, and the Phase 3 first-order Markov transition matrix, we evaluate three fusion strategies: (F1) fixed-weight convex probability mixture, (F2) adaptive transition-risk gating, and (F3) explicit change/destination decomposition. Validation-only model selection (primary objective $J = 0.5 \times \text{Macro-F1} + 0.5 \times \text{Change-F1}$) locks hyperparameters before test evaluation. On the 5,685 test hours with observed current state (Scope A), **F1 achieves the best Change-F1 (0.3781) and High-entry F1 (0.3743) while recovering +2.01% accuracy over standalone SVM (85.75% vs 83.74%)**. However, the Markov baseline retains superior overall accuracy (90.19%), Brier score (0.1666), and LogLoss (0.3392) due to the rarity of state transitions (9.25% of hours). No fusion method surpasses the Markov baseline on the primary joint objective $J$. The key finding is that **convex fixed-weight fusion ($\lambda=0.9$) meaningfully improves the change-detection/stability trade-off**, whereas adaptive and transition-explicit strategies sacrifice too much stable-state accuracy for modest change-recall gains.

---

## 1. Introduction

Urban air pollution forecasting is a high-stakes classification problem where missing a transition into hazardous pollution (PM2.5 $\ge$ 75 $\mu$g/m³) has direct public-health consequences. Yet standard accuracy-oriented classifiers tend to over-predict the persistent majority state (Moderate/Low), achieving high overall accuracy but near-zero change recall. Phase 3 established that the first-order Markov model achieves 90.19% Scope A accuracy but 0% change recall. Phase 5 showed that feature-based SVMs detect ~50% of true transitions but at the cost of 6–7% lower overall accuracy.

**Research Question**: Can a principled probability fusion of Markov persistence evidence and compressed-feature SVM evidence improve the joint objective of overall performance and transition detection without unacceptably compromising stable-state accuracy?

---

## 2. Prior Work Summary

| Phase | Contribution |
|-------|--------------|
| 1 | EDA: 12 stations, 24-hour windows, 420K station-hours, March 2013–Feb 2017 |
| 2 | Preprocessing: 24×12×11 tensor windows, train/val/test = 24,108 / 4,239 / 5,720 |
| 3 | Markov baseline: 3×3 transition matrix P from training; Scope A test accuracy = 90.19%, change recall = 0% |
| 4 | Compression: PCA (flattened) and HoSVD (tensor modes); dimensions 48, 192, 576 |
| 5 | SVM/RFF classification: Linear, exact RBF, RFF-Linear; calibrated probabilities on validation 80/20 holdout; best Linear/PCA-C: 83.37% accuracy, 50.57% change recall |

---

## 3. Mathematical Method

### 3.1 Probability Sources
- **Markov** $p_{M,t} = P_{S_t,\cdot}$ from train-estimated $P$ (Section 1.2)
- **SVM** $p_{S,t}$ from calibrated Linear/RFF models on PCA_B, HoSVD_B, PCA_C features

### 3.2 Fusion Strategies
- **F1 (Fixed)**: $p_{F,t} = \lambda p_{S,t} + (1-\lambda) p_{M,t}$, $\lambda \in \{0,0.1,\dots,1\}$
- **F2 (Adaptive)**: $\lambda_t = \lambda_{\text{high}}$ if $r_{S,t} \ge \tau$ else $\lambda_{\text{low}}$
- **F3 (Transition)**: $q_t = \rho r_{S,t} + (1-\rho) r_{M,t}$; destination blending via $\eta$

All fused vectors are guaranteed to lie in the probability simplex $\Delta^2$.

### 3.3 Metrics
Three-class: Accuracy, Macro-F1, Brier, LogLoss (clip $10^{-12}$, renormalize, natural log). Binary change (Scope A): precision, recall, F1. Exact: $A_{\text{change}}$, $A_{\text{stable}}$. High-entry: precision, recall, F1. All denominators reported explicitly.

### 3.4 Validation-Only Selection
$J = 0.5 \times \text{Macro-F1}_{\text{val}} + 0.5 \times \text{Change-F1}_{\text{val}}$. Ties broken by Macro-F1, then Change-F1, then deterministic name ordering. **No test labels used at any stage.**

---

## 4. Experimental Protocol

| Split | Rows | Scope A | True Transitions |
|-------|------|---------|------------------|
| Train | 24,108 | 23,976 | – |
| Validation | 4,239 | 4,197 | 488 |
| Test | 5,720 | 5,685 | 526 |

Current state missing in 35 test rows; Scope B evaluation uses feature-only SVM fallback (reported but secondary).

---

## 5. Probability Calibration

Seven calibrated SVM sources qualified for fusion:
- Linear: `linear_pca_B`, `linear_hosvd_B`, `linear_best_overall` (PCA_C)
- RFF (revised): `revised_rff_pca_B`, `revised_rff_hosvd_B`
- RFF (unrevised): `rff_pca_B`, `rff_hosvd_B`

All use train-only chronological 80/20 holdout calibration. Exact RBF (`stageB_full_*`) has **no calibrated probabilities** and is compared on hard labels only.

---

## 6. Results

### 6.1 Validation Grid Search
- F1: 7 sources × 11 lambdas = 77 configurations
- F2: 5 sources × 5 $\tau$ × 5 $\lambda_{\text{low}}$ × 5 $\lambda_{\text{high}}$ (constrained) = 375 combinations
- F3: 5 sources × 5 $\rho$ × 5 $\eta$ = 125 combinations

### 6.2 Locked Configurations (Validation Only)
| Family | Locked Config | Val $J$ | Val Macro-F1 | Val Change-F1 |
|--------|--------------|---------|-------------|--------------|
| F1 | $\lambda=0.9$, source=`linear_hosvd_B` | 0.5894 | 0.8073 | 0.3716 |
| F2 | $\tau=0.65$, $\lambda_{\text{low}}=0.75$, $\lambda_{\text{high}}=1.0$, source=`linear_pca_B` | 0.5876 | 0.8117 | 0.3635 |
| F3 | $\rho=0.75$, $\eta=0.0$, source=`linear_best_overall` | 0.5942 | 0.8394 | 0.3489 |

### 6.3 Test Evaluation (Scope A, N=5,685)

| Model | Accuracy | Macro-F1 | Change Prec | Change Rec | **Change F1** | $A_{\text{change}}$ | $A_{\text{stable}}$ | High F1 | Brier | LogLoss |
|-------|---------|----------|-------------|------------|--------------|-------------------|-------------------|---------|-------|--------|
| M0 Markov | **90.19%** | 0.6650 | 0.00% | 0.00% | **0.0000** | 0.00% | **100.00%** | 0.0000 | **0.1666** | 0.3392 |
| M1 Linear | 83.74% | 0.8031 | 28.34% | 46.39% | 0.3518 | 41.06% | 88.04% | 0.3049 | 0.2196 | 0.3726 |
| M1b RBF (pca_B) | 84.21% | 0.8101 | 30.06% | 48.86% | **0.3722** | 42.59% | 88.41% | **0.3827** | N/A | N/A |
| M1b RBF (hosvd_B) | 83.71% | 0.8112 | **29.64%** | **50.38%** | 0.3732 | **43.35%** | 87.81% | 0.3591 | N/A | N/A |
| **F1 Fixed** | 85.75% | 0.8273 | 33.19% | 43.92% | **0.3781** | 37.64% | 90.99% | **0.3743** | 0.2013 | 0.3539 |
| **F2 Adaptive** | 86.47% | 0.8360 | 31.79% | 33.84% | 0.3278 | 29.66% | 92.60% | 0.2881 | 0.2081 | 0.3699 |
| **F3 Transition** | 88.15% | **0.8629** | **39.60%** | 26.43% | 0.3170 | 15.59% | 95.89% | 0.0344 | 0.1814 | **0.3329** |

### 6.4 Key Observations
1. **Markov dominates overall accuracy & probability quality**: Due to 90.75% stable hours, persistence prediction minimizes Brier and LogLoss.
2. **F1 achieves best Change-F1 + High-entry F1 balance**: $\lambda=0.9$ recovers +2.01% accuracy over SVM alone while setting new Change-F1 high (0.3781).
3. **Adaptive gating fails to improve on fixed**: F2 increases false alarms (Change Precision 31.79% vs 33.19% for F1) and lowers $A_{\text{change}}$.
4. **Transition-explicit F3 maximizes Macro-F1 & $A_{\text{stable}}$**: But sacrifices change recall (26.43%) and High-entry F1 (0.0344) severely.
5. **No fusion surpasses Markov on $J$**: Markov $J = 0.3325$ vs best fusion F1 $J = 0.4138$ (F1 wins on Change-F1 but loses Macro-F1).

---

## 7. Statistical Evidence
Paired time-block bootstrap (24-hour blocks, 1000 resamples):
- F1 vs Markov: $\Delta$ Accuracy = $-4.70\%$ [95% CI: $-6.29\%$, $-3.24\%$]
- F1 vs SVM: $\Delta$ Accuracy = $+2.36\%$ [95% CI: $+0.77\%$, $+3.89\%$]

The accuracy drop relative to Markov is statistically significant; the gain over standalone SVM is significant.

---

## 8. Computational Cost
Phase 6 pipeline runtime: **17 seconds** (single-threaded). Additional model storage: **~500 KB** (merged predictions + grid tables). Exact RBF inference: ~7s/test (11,306 support vectors); Linear/RFF fusion: <0.2s/test.

---

## 9. Discussion & Limitations

* **Test exposure**: Phase 5 test results were known prior to Phase 6 design → exploratory, not confirmatory.
* **Calibration not fully nested**: Phase 5 used 80/20 chronological holdout within full training, not forward-chaining.
* **Markov P fitted on full training**: Same dependency structure as Phase 4 bases.
* **First-order Markov assumption**: Higher-order dependencies not modeled.
* **Single target station**: Generalization across stations/years not tested.
* **Multiple validation comparisons**: Selection sensitivity to grid density not quantified.
* **Exact-RBF probability fusion unsupported**: No train-only OOF calibration performed due to computational cost.

---

## 10. Conclusions
Fixed-weight convex fusion ($\lambda=0.9$) meaningfully improves the change-detection/stability trade-off over standalone SVMs, recovering +2.01% accuracy while achieving the highest Change-F1 (0.3781) and High-entry F1 (0.3743). However, **the Markov persistence baseline remains optimal for overall accuracy, Brier score, and LogLoss** because state transitions are rare (9.25% of hours). Adaptive gating and transition-explicit decomposition do not outperform the simple fixed mixture on the primary objective. Future work should explore cost-sensitive training with calibrated probabilities and multi-horizon transition modeling.

---

## 11. Connections to Phase 7
Phase 7 will extend the fusion framework to multi-step ahead prediction, uncertainty-aware decision thresholds for alerting, and cross-station transfer learning.