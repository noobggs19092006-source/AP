# FINAL PROJECT TECHNICAL HANDOVER AND COMPREHENSIVE RESEARCH AUDIT

**Project Title:** Air Pollution Prediction Using Tensor Compression, Support Vector Machines, Markov Chains, and Transition-Precursor-Aware Fusion  
**Target Publication:** IEEE Transactions / IEEE Conference on Signal Processing and Environmental Data Analytics  
**Auditor & Research Analyst:** Technical Project Audit Engine  
**Handover Date:** November 23, 2026  
**Repository Reference:** `https://github.com/noobggs19092006-source/AP`  
**Working Environment:** Windows 10/11 x64, Python 3.11.0, NumPy 1.26.4, Scikit-Learn 1.9.1  

---

## SECTION A — EXECUTIVE PROJECT OVERVIEW

This technical handover delivers the comprehensive mathematical, algorithmic, empirical, and architectural evidence for our university research project: **Air Pollution Prediction Using Tensor Compression (PCA/HoSVD), Kernel Support Vector Machines, Random Fourier Features, Markov Chains, and Transition-Precursor-Aware Fusion**.

The central challenge in computational air-quality forecasting is that severe pollution state transitions (e.g., rapid degradation into hazardous particulate concentrations) are statistically rare events embedded within long stretches of persistence. In the UCI Beijing multi-site dataset (35,064 hourly records across 12 monitoring stations, 2013–2017), **90.75% of consecutive hourly samples remain in the same pollution category**, while only **9.25% undergo an abrupt state change**.

Standard statistical and machine learning models fail on this task in opposite ways:
1. **The Markov Persistence Blind Spot:** A first-order, three-state Markov transition model achieves high overall accuracy (**90.75%**) and perfect stable-state accuracy (**100.00%**), but in hard categorical predictions, it detects **0 out of 526 actual state transitions (0.00% recall)**. It functions purely as a persistence model.
2. **The SVM Precision/False Alarm Trade-off:** Standalone Support Vector Machines trained on compressed historical spatio-temporal tensor representations detect transition dynamics (**46.39% recall** for Linear SVM), but suffer from substantial false alarms (616 false transitions), dragging overall classification accuracy down to **83.69%**.
3. **The Phase 6 Destination Fusion Gap:** Phase 6 combined Markov transition probabilities with calibrated SVM probabilities under a change-and-destination framework (F3 transition fusion). While F3 maintained relatively high accuracy (**88.46%**), its transition recall was severely constrained (**26.43%**, detecting only 139 of 526 changes).

The missing **Phase 7 Transition-Precursor Implementation** was designed to test the following research hypothesis:
> *Can physically motivated early-warning signals—specifically threshold proximity, local temporal rate of change, and cross-station spatial pollution surges—improve next-hour pollution state change detection when integrated into the Markov–SVM fusion framework?*

### Verified Finding Summary:
The validation-selected Phase 7 model (**E4**: threshold proximity + target-station trends fused with F3 at $\alpha = 0.75$) successfully doubles transition sensitivity:
- **Change Recall:** Jumps from **26.43%** (Phase 6 F3) to **55.89%** (E4) — an absolute gain of **+29.46 percentage points** (95% CI: [+23.98%, +35.18%], $p < 0.001$).
- **Exact Destination Accuracy:** Rises from **15.59%** to **40.11%** (gain of **+24.52 percentage points**, 95% CI: [+19.62%, +29.50%]).
- **High-Pollution Entry F1:** Increases from **0.0344** to **0.3026** (recall jumps from 4.9% to 21.3%).
- **Documented Trade-off:** This transition sensitivity gain comes at an empirical cost: false alarms increase from 212 to 622, overall accuracy declines from **88.46% to 83.52%** ($-4.94$ pp), and stable-state accuracy falls from **95.89% to 87.94%** ($-7.95$ pp).

All deliverables, 20 publication figure categories, 6 vector diagrams, unit tests (34/34 passing), and independent numerical audits (163 PASS, 0 FAIL) are verified and packaged for IEEE paper drafting.

---

## SECTION B — REPOSITORY AND FILE INVENTORY

A complete audit of all repository files confirms that all historical artifacts from Phases 2–6 and the prior nested calibration experiment have been frozen and preserved byte-for-byte. All binary `.npz`, `.pkl`, and `.zip` files were confirmed as **real binary files (not Git LFS pointers)**.

| Phase | Phase Name & Directory | Main Purpose / Algorithm | Source Code Path | Input Artifacts | Output Artifacts & Models | Verification Status | Completion |
|---|---|---|---|---|---|:---:|:---:|
| **Phase 1** | Exploratory Data Analysis | Initial data inspection & distributions | *Implicit in raw CSVs* | `beijing+multi+site+air+quality+data/` | *No standalone Phase 1 dir* | MISSING (Implicit) | Partial |
| **Phase 2** | Preprocessing & Windowing (`Phase 2/`) | 24-hr tensor construction ($24 \times 12 \times 11$), normalization | `Phase 2/phase2_build.py` | 12 PRSA Station CSVs | `phase2_output/*.npz`, `phase2_summary.json` | VERIFIED | Complete |
| **Phase 3** | Markov Chain (`Air_Pollution_Phase3_Results/`) | 3-state row-stochastic transition modeling | `markov_core.py`, `phase3_markov.py` | `Phase 2/phase2_output/` | `markov_transition_matrix.csv`, `markov_predictions_test.csv` | VERIFIED | Complete |
| **Phase 4** | Tensor Compression (`phase4_output/`) | Mode-wise HoSVD vs Matrix PCA ($3168 \to 48, 192, 576$) | `phase4_pca_hosvd.py` | `Phase 2/phase2_output/*.npz` | `features_*_pca_*.npz`, `features_*_hosvd_*.npz`, `pca_hosvd_comparison.csv` | VERIFIED | Complete |
| **Phase 5** | SVM / Kernel / RFF (`Phase5_SVM_Kernel_RFF_Results/`) | Linear SVM, Exact RBF SVM, Random Fourier Features | `source/*.py` (40 fits linear, 36 RBF, 30 RFF) | `phase4_output/*.npz` | `models/*.pkl`, `final_evaluation_scopeAB.csv`, `audit_summary.json` | VERIFIED | Complete |
| **Phase 6** | Markov–SVM Fusion (`Phase6_Markov_SVM_Fusion_Results/`) | F1 Fixed, F2 Adaptive, F3 Transition/Destination Blending | `source/*.py`, `config/*.json` | Phase 3 matrix, Phase 5 models | `predictions/test/*.csv`, `final_evaluation.csv` | VERIFIED | Complete |
| **Phase 6 Ver**| Phase 6 Audit (`Phase6_Independent_Verification_Results/`) | Independent replication of Phase 6 fusion | `audit_*.py` | Phase 6 outputs | `audit_summary.json` | VERIFIED | Complete |
| **Phase 7 Cal**| Nested CV Calibration (`Phase7_Nested_CV_Calibration/`) | 5-fold OOF Platt scaling & cost-sensitive weights | `source/nested_calibration.py` | `features_*_pca_B.npz` | `results/phase7_comparison.csv`, `phase7_summary.json` | AUDITED (Scope B labeled Scope A) | Complete |
| **Phase 7 Pre**| Transition Precursor Results (`Phase7_Transition_Precursor_Results/`) | Threshold proximity, temporal rates, spatial trends + F3 fusion | `source/*.py`, `phase7_run.py`, `phase7_publish.py` | Phase 2 tensors, Phase 3 matrix, Phase 5 Linear Best | 18 logistic models, 16 prediction CSVs, 20 figures, 6 diagrams | VERIFIED | Complete |
| **Phase 7 Aud**| Phase 7 Independent Verification (`Phase7_Independent_Verification_Results/`) | Independent mathematical recomputation & paired bootstrap | `audit_source/*.py`, `phase7_audit_run.py` | Phase 7 prediction CSVs & raw data | `results/*.csv`, 163 PASS audit matrix, test logs | VERIFIED | Complete |

---

## SECTION C — COMPLETE PHASE 1–6 TECHNICAL BACKGROUND

### 1. Phase 1 — Dataset and Exploratory Data Analysis
- **Source:** UCI Beijing Multi-Site Air Quality Dataset (March 1, 2013 to February 28, 2017).
- **Spatial Coverage:** 12 air monitoring stations across Beijing: *Aotizhongxin (target station 0)*, *Changping*, *Dingling*, *Dongsi*, *Guanyuan*, *Gucheng*, *Huairou*, *Nongzhanguan*, *Shunyi*, *Tiantan*, *Wanliu*, *Wanshouxigong*.
- **Temporal Resolution:** Hourly observations ($N_{raw} = 35,064$ common hours per station).
- **Physical Variables (11 numerical channels):** $\text{PM}_{2.5}$, $\text{PM}_{10}$, $\text{SO}_2$, $\text{NO}_2$, $\text{CO}$, $\text{O}_3$, $\text{TEMP}$, $\text{PRES}$, $\text{DEWP}$, $\text{RAIN}$, $\text{WSPM}$.
- **Target Categorization (State $S_t$):** Instantaneous PM2.5 categories based on project definitions:
  $$S_t = \begin{cases} 0 \text{ (Low)}, & m_t < 35 \,\mu\text{g/m}^3 \\ 1 \text{ (Moderate)}, & 35 \le m_t < 75 \,\mu\text{g/m}^3 \\ 2 \text{ (High)}, & m_t \ge 75 \,\mu\text{g/m}^3 \end{cases}$$
- **Data Reality:** Missing values occur due to sensor telemetry outages. Phase 1 exploration revealed severe autocorrelation in PM2.5, seasonal winter surges (coal heating), and cross-station correlation driven by synoptic wind patterns.

### 2. Phase 2 — Preprocessing Pipeline and Tensor Construction
- **Forward Filling:** Limited forward filling of up to 6 consecutive hours within each chronological split independently.
- **Median Imputation:** Remaining missing values replaced using **training-set derived medians only** (preventing forward data leakage).
- **Target Integrity:** Prediction targets $y_t = S(m_{0, t+1})$ and observed current states $S_t = S(m_{0, t})$ are strictly derived from **unfilled, original raw PM2.5 observations**. If $m_{0, t}$ was missing in raw data, $S_t = -1$.
- **Tensor Windowing:** Each sample at hour $t$ spans 24 hours of history: $X_t \in \mathbb{R}^{24 \times 12 \times 11}$.
- **Flattened Input Dimension:** $D = 24 \times 12 \times 11 = 3,168$ features.
- **Chronological Split Allocation:**
  - *Training:* 2013-03-01 00:00 to 2015-12-31 23:00 ($N = 24,108$ usable samples; 732 missing target excluded; 132 missing $S_t$).
  - *Validation:* 2016-01-01 00:00 to 2016-06-30 23:00 ($N = 4,239$ usable samples; 105 missing target excluded; 42 missing $S_t$).
  - *Test:* 2016-07-01 00:00 to 2017-02-28 23:00 ($N = 5,720$ usable samples; 88 missing target excluded; 35 missing $S_t$).
- **Normalization:** Per-station, per-channel z-score standardization fit strictly on the training set: $z = (m - \mu) / \sigma$.

### 3. Phase 3 — First-Order Markov Chain Modeling
The Markov baseline assumes $P(S_{t+1} = j \mid S_t, S_{t-1}, \dots) = P(S_{t+1} = j \mid S_t)$. Empirical transition counts on the training set yielded the row-stochastic matrix $P \in \mathbb{R}^{3 \times 3}$:
$$P = \begin{pmatrix} 0.912388 & 0.085824 & 0.001788 \\ 0.106478 & 0.769297 & 0.124225 \\ 0.005512 & 0.063721 & 0.930768 \end{pmatrix}$$
- **Persistence Behavior:** The diagonal dominance ($P_{00}=0.9124, P_{11}=0.7693, P_{22}=0.9308$) reflects high atmospheric inertia.
- **The Blind Spot:** In hard decision making ($\hat{y}_t = \arg\max_j P_{S_t, j}$), the diagonal probability is always the largest. Consequently, the Markov chain **never predicts a state transition**, achieving 100% stable accuracy but 0.00% change recall.

### 4. Phase 4 — Dimensionality Reduction: PCA vs HoSVD
- **Matrix PCA:** SVD of mean-centered flattened training matrix $X \in \mathbb{R}^{N \times 3168}$: $X = U \Sigma V^T$. Subspace projection: $z = V_k^T x$.
- **Tensor HoSVD:** Mode-wise unfolding of 3rd-order tensor $\mathcal{X} \in \mathbb{R}^{24 \times 12 \times 11}$. Left singular matrices computed via eigendecomposition of mode Gram matrices:
  - Mode-1 (Time, 24): $U^{(1)} \in \mathbb{R}^{24 \times r_1}$
  - Mode-2 (Station, 12): $U^{(2)} \in \mathbb{R}^{12 \times r_2}$
  - Mode-3 (Feature, 11): $U^{(3)} \in \mathbb{R}^{11 \times r_3}$
  - Core tensor: $\mathcal{S} = \mathcal{X} \times_1 U^{(1)T} \times_2 U^{(2)T} \times_3 U^{(3)T}$.
- **Comparative Findings (Config B, $k=192$):**
  - PCA-B ($k=192$): Test relative reconstruction error $= 0.2119$, retained energy $= 94.62\%$.
  - HoSVD-B ($r_1=8, r_2=4, r_3=6$, core size 192): Test relative reconstruction error $= 0.3394$.
  - *Theoretical Insight:* PCA achieves strictly lower reconstruction error for the same scalar feature count because it solves an unconstrained global rank-$k$ optimization, whereas HoSVD is constrained to a multi-linear rank-$(r_1, r_2, r_3)$ separable structure. However, HoSVD retains explicit temporal and spatial mode factorizations.

### 5. Phase 5 — Support Vector Classification (Linear, Kernel, RFF)
- Evaluated linear SVM, exact RBF kernel SVM, and Random Fourier Features (RFF) on PCA and HoSVD representations.
- **Best Model Overall:** Standalone Linear SVM on PCA-C ($k=576$, $C=0.1$) with calibrated probabilities.
  - Achieved Scope A test accuracy of **83.69%**, macro-F1 of **0.8023**, and change recall of **46.39%**.
  - While transition recall improved from 0% to 46.39%, overall accuracy fell by 7.06 percentage points due to 616 false alarms.
- **RFF vs Exact RBF:** Exact RBF on HoSVD-B achieved validation macro-F1 of $0.7604$ with 4,033 support vectors. RFF with $D=2048$ achieved validation macro-F1 of $0.7465$ with $5.2\times$ faster training and $O(1)$ constant test evaluation time independent of support vector count.

### 6. Phase 6 — Markov–SVM Fusion
Phase 6 introduced probability blending between calibrated SVM probabilities $p_t^{SVM}$ and Markov row probabilities $P_{S_t}$:
- **F1 (Fixed Fusion):** $p_t^{F1} = \lambda p_t^{SVM} + (1-\lambda) P_{S_t}$. Selected $\lambda = 0.9$.
- **F2 (Adaptive Fusion):** High-confidence thresholding on $\tau = 0.65$.
- **F3 (Transition/Destination Fusion):** Separated transition occurrence from destination state:
  $$q_t^{F3} = \rho (1 - p_t^{SVM}(S_t)) + (1-\rho) (1 - P_{S_t, S_t})$$
  $$d_t^{F3}(j) = \frac{\eta p_t^{SVM}(j) + (1-\eta) P_{S_t, j}}{\sum_{k \ne S_t} [\eta p_t^{SVM}(k) + (1-\eta) P_{S_t, k}]}, \quad j \ne S_t$$
  - Selected Parameters: $\rho = 0.75$, $\eta = 0.0$ (SVM determines transition risk; pure Markov determines destination distribution).
  - Scope A Test Performance: Accuracy $= 88.46\%$, macro-F1 $= 0.8661$, change recall $= 26.43\%$, exact destination accuracy $= 15.59\%$.
  - Limitation: Constrained change recall because F3 lacked explicit precursor features signaling imminent boundary crossings.

---

## SECTION D — EXISTING PHASE 7 NESTED CALIBRATION AUDIT

The repository contains an earlier experiment in `Phase7_Nested_CV_Calibration/`. A rigorous code and metric audit confirms that this experiment is **NOT the missing transition-precursor Phase 7** and is **NOT directly comparable** to the true Scope A benchmarks.

### Independent Audit Findings:
1. **Target of Experiment:** 5-fold cross-validation Platt scaling for Exact RBF SVMs and cost-sensitive retraining weights.
2. **In-Sample Calibrator Fitting:** Code inspection of `source/nested_calibration.py` (lines 60–70) reveals that the one-vs-rest logistic calibrators are fitted using:
   ```python
   clf.fit(train_scores[:, c].reshape(-1, 1), y_binary)
   ```
   where `train_scores = svm.decision_function(X_train_s)`. This represents in-sample decision score calibration rather than strictly out-of-fold calibration fitting.
3. **Population & Scope Discrepancy (Scope B mislabeled as Scope A):**
   - The summary comparison table (`phase7_comparison.csv`) reports Markov baseline accuracy as **0.904196**.
   - Verification: On all 5,720 Scope B rows (including 35 missing $S_t$ samples evaluated with zero persistence), Markov accuracy is $5,172 / 5,720 = \mathbf{0.9041958}$. On true Scope A ($N=5,685$), Markov accuracy is **0.907476**.
   - Thus, the nested calibration experiment evaluated on all 5,720 Scope B rows while labeling the column as Scope A.
4. **Comparability Classification:** **NOT DIRECTLY COMPARABLE.**
   - Lacks row-level prediction CSVs.
   - Evaluated on a different effective sample population.
   - Does not address threshold proximity, temporal rates, or cross-station precursor features.
   - Preserved as an independent prior experiment without modification.

---

## SECTION E — COMPLETE PHASE 7 TRANSITION-PRECURSOR IMPLEMENTATION

The final Phase 7 precursor experiment was implemented in `Phase7_Transition_Precursor_Results/` as a reproducible extension of the Phase 6 F3 framework.

### 1. Physical PM2.5 Reconstruction
Historical standardized tensor inputs $z_{s, t} \in \mathbb{R}$ are converted back to physical concentration units ($m_{s, t} \in \mu\text{g/m}^3$) using the verified Phase 2 training normalization parameters:
$$m_{s, t} = z_{s, t} \cdot \sigma_s + \mu_s$$
where $\sigma_s$ and $\mu_s$ are station-specific training parameters (for Aotizhongxin, $\mu_0 = 84.994, \sigma_0 = 81.129$). Reconstructed values match raw station records with error $< 10^{-11}$.

### 2. Feature Formulation (Three Precursor Groups)
- **Group A — Threshold Proximity ($h_t^{threshold} \in \mathbb{R}^6$):**
  - Absolute distances: $d_{35, t} = |m_{0, t} - 35|$, $d_{75, t} = |m_{0, t} - 75|$.
  - Minimum distance to boundary: $d_{min, t} = \min(d_{35, t}, d_{75, t})$.
  - Current state one-hot indicators: $\mathbf{1}[S_t = 0], \mathbf{1}[S_t = 1], \mathbf{1}[S_t = 2]$.
  - *Rationale:* Samples near $35$ or $75\,\mu\text{g/m}^3$ have higher physical susceptibility to crossing boundaries next hour.
- **Group B — Target Temporal Rates ($h_t^{temporal} \in \mathbb{R}^2$):**
  - 3-hour velocity: $v_{3, t} = \frac{m_{0, t} - m_{0, t-3}}{3}$ (index 23 minus index 20).
  - 6-hour velocity: $v_{6, t} = \frac{m_{0, t} - m_{0, t-6}}{6}$ (index 23 minus index 17).
  - *Rationale:* A sample at $33\,\mu\text{g/m}^3$ with $v_{3, t} = +8\,\mu\text{g/m}^3/\text{h}$ is almost certain to cross into Moderate; with $v_{3, t} = -4$, it will remain Low.
- **Group C — Cross-Station Regional Trends ($h_t^{cross} \in \mathbb{R}^3$):**
  - Derived from the 11 non-target stations ($s = 1, \dots, 11$): $v_{s, t} = \frac{m_{s, t} - m_{s, t-3}}{3}$.
  - Mean velocity: $v_{mean, t} = \frac{1}{11} \sum_{s=1}^{11} v_{s, t}$.
  - Maximum velocity: $v_{max, t} = \max_{s \ne 0} v_{s, t}$.
  - Fraction rising: $f_{rise, t} = \frac{1}{11} \sum_{s=1}^{11} \mathbf{1}[v_{s, t} > 0]$.
  - *Rationale:* Beijing air pollution is regional; wind shifts move pollution plumes across monitoring stations before reaching Aotizhongxin.

### 3. Binary Precursor Transition Classifier
A regularized logistic regression model is trained on training Scope A rows ($N=23,976$, changes $C_t = \mathbf{1}[y_t \ne S_t]$):
$$q_t^P = \sigma(w^T h_t + b) = \frac{1}{1 + e^{-(w^T h_t + b)}}$$
- Scaling: Standardized using `StandardScaler` fitted strictly on training Scope A features.
- Evaluated Grid: $C \in \{0.1, 1.0, 10.0\}$, class weighting $\in \{\text{None}, \text{balanced}\}$.
- Winner: $C = 1.0$, balanced class weighting. Balanced weighting addresses the $9:1$ stability-to-change imbalance.

### 4. Integration with Phase 6 F3 Architecture
The precursor change probability $q_t^P$ is fused with the Phase 6 F3 change probability $q_t^{F3}$:
$$q_t^{new} = \alpha q_t^P + (1 - \alpha) q_t^{F3}, \quad \alpha \in [0, 1]$$
Final 3-class probability synthesis:
$$p_t^{new}(S_t) = 1 - q_t^{new}$$
$$p_t^{new}(j) = q_t^{new} \cdot d_t^{F3}(j), \quad \forall j \ne S_t$$
where $d_t^{F3}(j)$ is the verified Phase 6 F3 conditional destination distribution ($\eta=0$, pure Markov destination). Probability simplex condition $\sum_{j=0}^2 p_t^{new}(j) = 1$ is mathematically guaranteed.

---

## SECTION F — PHASE 7 EXPERIMENTAL DESIGN

All experiments use the identical chronological partitions and Scope A/B definitions:
- **E0:** Original Markov Baseline.
- **E1:** Existing Standalone Linear SVM Reference (PCA-C, $k=576$).
- **E2:** Original Phase 6 F3 Transition Fusion ($\rho=0.75, \eta=0.0$).
- **E3:** F3 + Threshold-Proximity Features ($h^{threshold}$, 6 features).
- **E4:** F3 + Threshold & Target Trends ($h^{threshold} + h^{temporal}$, 8 features).
- **E5:** F3 + All Three Precursor Groups (11 features).
- **E6:** E5 with Validation-Selected Transition Decision Threshold $\theta = 0.40$.
- **E7:** Precursor-Only Change Probability ($\alpha = 1.0$) with F3 Destination Distribution.

**Validation Selection Protocol:** Model selection locked strictly on Validation Scope A ($N = 4,197$) using the Phase 6 harmonic objective:
$$J = 0.5 \cdot \text{MacroF1} + 0.5 \cdot \text{ChangeF1}$$
Winning configuration: **E4** (temporal rates, $C=1.0$, balanced, $\alpha=0.75$), achieving validation $J = 0.620206$.

---

## SECTION G — VERIFIED NUMERICAL RESULTS

The table below presents the verified numerical results across all controlled experiments on the hold-out test set under Scope A ($N = 5,685$, 526 actual transitions):

| Model ID | Experiment Description | Split | Scope | N_Samples | Accuracy | Macro-F1 | Change Precision | Change Recall | Change-F1 | Exact Destination | Stable Accuracy | High-Entry F1 | Verification Status |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **E0** | Markov Baseline | test | ScopeA | 5685 | 0.907476 | 0.889509 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | VERIFIED |
| **E1** | Standalone Linear SVM | test | ScopeA | 5685 | 0.836939 | 0.802332 | 0.283391 | 0.463878 | 0.351839 | 0.410646 | 0.880403 | 0.304933 | VERIFIED |
| **E2** | Phase 6 F3 Fusion | test | ScopeA | 5685 | 0.884609 | 0.866149 | 0.396011 | 0.264259 | 0.316990 | 0.155894 | 0.958907 | 0.034364 | VERIFIED |
| **E3** | F3 + Threshold Proximity | test | ScopeA | 5685 | 0.834653 | 0.805659 | 0.319172 | 0.557034 | 0.405817 | 0.401141 | 0.878852 | 0.304976 | VERIFIED |
| **E4** | **Selected Phase 7 Model** | test | ScopeA | 5685 | **0.835180** | **0.807418** | **0.320961** | **0.558935** | **0.407767** | **0.401141** | **0.879434** | **0.302632** | **VERIFIED** |
| **E5** | F3 + All 3 Groups | test | ScopeA | 5685 | 0.842392 | 0.817270 | 0.339286 | 0.577947 | 0.427567 | 0.422053 | 0.885249 | 0.334495 | VERIFIED |
| **E6** | E5 + Decision Theta (0.40) | test | ScopeA | 5685 | 0.843975 | 0.802350 | 0.337789 | 0.528517 | 0.412157 | 0.349810 | 0.894359 | 0.306040 | VERIFIED |
| **E7** | Precursor Only (alpha=1.0)| test | ScopeA | 5685 | 0.788391 | 0.759879 | 0.280297 | 0.716730 | 0.402993 | 0.553232 | 0.812367 | 0.327470 | VERIFIED |

*Scope B Results ($N=5,720$ including 35 missing current states with deterministic calibrated feature-only SVM fallback):*
- E0 (Scope B): Accuracy $= 0.907517$, Macro-F1 $= 0.889684$.
- E2 (Scope B): Accuracy $= 0.884790$, Macro-F1 $= 0.866463$.
- E4 (Scope B): Accuracy $= 0.835664$, Macro-F1 $= 0.808130$.

---

## SECTION H — PHASE 6 VERSUS PHASE 7 ANALYSIS

### 1. Direct Paired Comparison: Selected E4 vs Phase 6 F3 (E2)
- **Change Recall:** $+29.46$ percentage points gain ($55.89\%$ vs $26.43\%$). E4 detects **294 transitions** vs 139 for F3 ($+155$ transitions detected).
- **Exact Destination Accuracy:** $+24.52$ percentage points gain ($40.11\%$ vs $15.59\%$). E4 correctly predicts destination state for **211 real transitions** vs 82 for F3 ($+129$ correct destinations).
- **Change-F1:** $+0.0908$ improvement ($0.4078$ vs $0.3170$).
- **High-Pollution Entry F1:** $+0.2682$ improvement ($0.3026$ vs $0.0344$).
- **Overall Accuracy Cost:** $-4.94$ percentage points ($83.52\%$ vs $88.46\%$).
- **Stable-State Accuracy Cost:** $-7.95$ percentage points ($87.94\%$ vs $95.89\%$).
- **False Alarm Increase:** $+410$ false alarms ($622$ for E4 vs $212$ for F3).

### 2. Paired 24-Hour Block Bootstrap Confidence Intervals (1,000 Resamples)
- $\Delta \text{Change Recall: } [+\mathbf{23.98\%},\, +\mathbf{35.18\%}]$ ($p < 0.001$, zero excluded).
- $\Delta \text{Exact Destination: } [+\mathbf{19.62\%},\, +\mathbf{29.50\%}]$ ($p < 0.001$, zero excluded).
- $\Delta \text{Change-F1: } [+\mathbf{0.0511},\, +\mathbf{0.1316}]$ ($p < 0.001$, zero excluded).
- $\Delta \text{High-Entry F1: } [+\mathbf{0.2132},\, +\mathbf{0.3231}]$ ($p < 0.001$, zero excluded).
- $\Delta \text{Overall Accuracy: } [-\mathbf{6.05\%},\, -\mathbf{3.93\%}]$ (statistically significant drop).
- $\Delta \text{Stable Accuracy: } [-\mathbf{9.38\%},\, -\mathbf{6.58\%}]$ (statistically significant drop).

---

## SECTION I — INDEPENDENT VERIFICATION FINDINGS

The independent audit package (`Phase7_Independent_Verification_Results/`) executed independently without importing implementation metric functions.
- **Verification Scorecard:** 163 PASS, 0 FAIL, 4 WARNING, 3 NOT VERIFIED.
- **Prior Phase Immutability:** All 378 historical files verified identical via SHA-256 (`artifact_integrity.csv`).
- **Equation Verification:** Independent logistic sigmoid calculations agree with model outputs with error $< 10^{-10}$.
- **Simplex Verification:** Simplex constraint verified on all validation and test probability vectors.
- **Graph Verification:** All 20 figures independently verified against raw prediction CSVs with zero numerical discrepancy (`graph_verification_matrix.csv`).
- **Unit Tests:** 22/22 Phase 7 tests passing; 12/12 Audit tests passing.

---

## SECTION J — MATHEMATICAL REFERENCE FOR IEEE PAPER

1. **Matrix SVD & PCA Projection:**
   $$X = U \Sigma V^T, \quad z_k = V_k^T (x - \mu), \quad E_k = \frac{\sum_{i=1}^k \sigma_i^2}{\sum_{i=1}^D \sigma_i^2}$$
2. **Tensor HoSVD Modal Factorization:**
   $$\mathcal{S} = \mathcal{X} \times_1 U^{(1)T} \times_2 U^{(2)T} \times_3 U^{(3)T}, \quad \mathcal{X} \approx \mathcal{S} \times_1 U^{(1)} \times_2 U^{(2)} \times_3 U^{(3)}$$
3. **Random Fourier Feature Mapping:**
   $$z(x) = \sqrt{\frac{2}{D}} \left[ \cos(w_1^T x + b_1), \dots, \sin(w_D^T x + b_D) \right]^T, \quad w_i \sim \mathcal{N}(0, 2\gamma I), \quad b_i \sim \mathcal{U}[0, 2\pi]$$
4. **Transition-Precursor Fusion:**
   $$q_t^{new} = \alpha \cdot \sigma(w^T h_t + b) + (1 - \alpha) \cdot q_t^{F3}$$
   $$p_t^{new}(S_t) = 1 - q_t^{new}, \quad p_t^{new}(j) = q_t^{new} \cdot d_t^{F3}(j) \quad (j \ne S_t)$$
5. **Weighted Accuracy Identity:**
   $$\text{Accuracy} = (1 - \pi_{change}) \cdot A_{stable} + \pi_{change} \cdot A_{change}$$
   Checked on E4: $(0.907476)(0.879434) + (0.092524)(0.401141) = 0.835180$ (exact).

---

## SECTION K — COMPLETE GRAPH AND DIAGRAM MANIFEST

All 20 publication figures and 6 vector diagrams exist in PNG, PDF, and source CSV/JSON formats:
- **Essential Figures for Main Paper:**
  - `FIG-01`: `01_accuracy_and_macro_f1.png` (Section VIII)
  - `FIG-02`: `02_change_recall_across_models.png` (Section VIII)
  - `FIG-03`: `03_correct_destination_on_actual_transitions.png` (Section VIII)
  - `FIG-04`: `04_stability_versus_transition_recall.png` (Section IX)
  - `FIG-05`: `05_transition_precision_recall_and_f1.png` (Section VIII)
  - `FIG-06`: `06_selected_phase_7_versus_phase_6.png` (Section IX)
  - `FIG-10`: `10_controlled_e0_to_e7_ablations.png` (Section VIII)
  - `FIG-19`: `19_transitions_into_high_pollution.png` (Section VIII)
  - `FIG-20`: `20_research_progression_on_matching_test_rows.png` (Section IX)
- **Essential Diagrams:**
  - `DIAG-01`: `01_project.png` (Section II)
  - `DIAG-02`: `02_phase7.png` (Section V)
  - `DIAG-06`: `06_contribution.png` (Section II)
- Complete table with captions and paths available in `IEEE_FIGURE_AND_DIAGRAM_MANIFEST.csv`.

---

## SECTION L — FILE TRANSFER CHECKLIST

Organized into Priority A (Paper drafting), Priority B (Audit verification), and Priority C (Supporting).
See `IEEE_REQUIRED_FILE_TRANSFER_CHECKLIST.md` for exact file-by-file paths.
Both archives are intact:
- `Phase7_Transition_Precursor_Results.zip` (11.62 MB)
- `Phase7_Independent_Verification_Results.zip` (84.6 KB)

---

## SECTION M — LITERATURE AND NOVELTY REVIEW

15 authentic papers verified in `literature/verified_reference_matrix.csv`:
- Deep Learning & Spatial Forecasting: Bekkar et al. (2021), Zhang et al. (2017).
- Markov Environmental Models: Zakaria et al. (2019).
- Foundations: Cortes & Vapnik (1995), Rahimi & Recht (2007), De Lathauwer et al. (2000), Kolda & Bader (2009).
- Probability Calibration & Evaluation: Niculescu-Mizil & Caruana (2005), Guo et al. (2017), Saito & Rehmsmeier (2015).
- Statistical Inference & Software: Fan et al. (2008), Chang & Lin (2011), Kunsch (1989), Jolliffe & Cadima (2016), Cawley & Talbot (2010).
- **Novelty Statement:** While threshold features, logistic regression, and Markov chains are established tools, their specific integration into a two-stage change-and-destination probability fusion framework to resolve environmental persistence blind spots is the verified contribution of this work.

---

## SECTION N — IEEE PAPER SECTION-BY-SECTION CONTENT MAP

Complete 11-section blueprint detailing required topics, mathematical formulations, evidence tables, recommended figures, safe claims, and claims to avoid is provided in `IEEE_PAPER_SECTION_CONTENT_MAP.md`.

---

## SECTION O — MISSING INFORMATION, LIMITATIONS AND RISKS

Transparent disclosure of four warnings and three unverified items:
1. Probability calibration degrades under balanced logistic weighting (ECE increases from 0.0570 to 0.0682).
2. Historical nested CV experiment used Scope B rows and in-sample Platt fitting; excluded from Scope A leaderboard.
3. Legacy precursor source files preserved but unused.
4. Künsch (1989) bootstrap reference verified at metadata level; full-text review recommended.
5. Authoritative original professor rubric not supplied (working descriptions satisfied).
6. Raw 3,168-feature SVM baseline absent from historical artifacts (PCA-C used as upper bound).
7. Row-level predictions for nested CV absent.
See `UNRESOLVED_ISSUES_AND_MISSING_EVIDENCE.md` for complete details.

---

## SECTION P — FINAL READINESS ASSESSMENT

**AUDIT CONCLUSION: READY FOR IEEE PAPER PREPARATION.**
The technical handover is complete. Every empirical claim is supported by verified CSVs, mathematical equations match executable code, figures match plotted numbers, and prior phases remain completely untouched. The downstream AI assistant has all necessary materials to produce the final IEEE paper.
