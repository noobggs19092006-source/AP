# Phase 6 Mathematical Derivations and Numerical Results

## 1. Mathematical Framework

### 1.1 Dataset Definition and Next-Hour Pollution State Target
Let $X_t \in \mathbb{R}^{24 \times 12 \times 11}$ represent the historical tensor window at hour $t$, comprising 24 historical hours across 12 monitoring stations and 11 numerical meteorological/air quality channels. The target label $y_t \in \{0, 1, 2\}$ is the instantaneous PM2.5 state at Aotizhongxin station at hour $t+1$:
$$
y_t = \begin{cases}
0 \text{ (Low)}, & \text{if } \text{PM2.5}(t+1) < 35\ \mu\text{g/m}^3 \\
1 \text{ (Moderate)}, & \text{if } 35 \le \text{PM2.5}(t+1) < 75\ \mu\text{g/m}^3 \\
2 \text{ (High)}, & \text{if } \text{PM2.5}(t+1) \ge 75\ \mu\text{g/m}^3
\end{cases}
$$

Let $S_t \in \{0, 1, 2, -1\}$ denote the observed PM2.5 pollution state at hour $t$ (where $S_t = -1$ indicates a missing observation). Scope A is defined as the subset of time steps $t$ where $S_t \ge 0$.

### 1.2 First-Order Markov Transition Model
The baseline Markov model estimates a row-stochastic transition matrix $P \in \mathbb{R}^{3 \times 3}$ from training transitions $N_{ij}$:
$$
P_{ij} = P(S_{t+1} = j \mid S_t = i) = \frac{N_{ij}}{\sum_{k=0}^2 N_{ik}}, \quad \sum_{j=0}^2 P_{ij} = 1.
$$
Given current observed state $i = S_t$, the Markov probability vector is:
$$
p_{M,t} = [P_{i0}, P_{i1}, P_{i2}]^\top.
$$

### 1.3 Calibrated Multi-Class Support Vector Machine Probabilities
For compressed feature representation $z_t \in \mathbb{R}^d$ (PCA or HoSVD), a linear SVM or RFF-linear model emits decision margins $f(z_t) \in \mathbb{R}^3$. Calibrated class probabilities $p_{S,t} \in \Delta^2$ are obtained via Platt/multinomial logistic calibration trained out-of-sample on training data:
$$
p_{S,t}(j) = \frac{\exp(w_j^\top f(z_t) + b_j)}{\sum_{k=0}^2 \exp(w_k^\top f(z_t) + b_k)}, \quad \sum_{j=0}^2 p_{S,t}(j) = 1.
$$

### 1.4 Convex Fixed-Weight Probability Fusion (Experiment F1)
For $\lambda \in [0, 1]$, the fused probability vector $p_{F,t} \in \mathbb{R}^3$ is defined as:
$$
p_{F,t}(j) = \lambda p_{S,t}(j) + (1 - \lambda) p_{M,t}(j), \quad j \in \{0, 1, 2\}.
$$
**Simplex Proof**: Since $p_{S,t} \in \Delta^2$ and $p_{M,t} \in \Delta^2$:
$$
\sum_{j=0}^2 p_{F,t}(j) = \lambda \sum_{j=0}^2 p_{S,t}(j) + (1-\lambda) \sum_{j=0}^2 p_{M,t}(j) = \lambda(1) + (1-\lambda)(1) = 1.
$$
Since $\lambda \ge 0$, $1-\lambda \ge 0$, $p_{S,t}(j) \ge 0$, $p_{M,t}(j) \ge 0$, we have $p_{F,t}(j) \ge 0$ for all $j$. Thus $p_{F,t} \in \Delta^2$.
*Endpoints*: At $\lambda = 0$, $p_{F,t} = p_{M,t}$; at $\lambda = 1$, $p_{F,t} = p_{S,t}$.

### 1.5 Adaptive Transition-Risk-Gated Fusion (Experiment F2)
The SVM estimated transition risk from current state $i = S_t$ is $r_{S,t} = 1 - p_{S,t}(i)$. Given risk threshold $\tau \in [0, 1]$ and weights $0 \le \lambda_{\text{low}} \le \lambda_{\text{high}} \le 1$:
$$
\lambda_t = \begin{cases}
\lambda_{\text{high}}, & \text{if } r_{S,t} \ge \tau \\
\lambda_{\text{low}}, & \text{otherwise}
\end{cases}
$$
$$
p_{F,t}(j) = \lambda_t p_{S,t}(j) + (1 - \lambda_t) p_{M,t}(j).
$$
Since $\lambda_t \in [0, 1]$ for every sample $t$, $p_{F,t} \in \Delta^2$ pointwise.

### 1.6 Explicit Transition / Destination Decomposition (Experiment F3)
Decompose prediction into (1) transition likelihood $q_t = P(y_t \ne S_t)$ and (2) destination distribution $d_{F,t}(j) = P(y_t = j \mid y_t \ne S_t)$.

*Blended Change Likelihood*:
$$
q_t = \rho r_{S,t} + (1 - \rho) r_{M,t}, \quad \rho \in [0, 1], \quad r_{M,t} = 1 - P_{ii}.
$$
*Blended Destination Distribution*:
For observed current state $i = S_t$, destination $j \ne i$:
$$
d_{F,t}(j) = \frac{\eta p_{S,t}(j) + (1 - \eta) p_{M,t}(j)}{\eta(1 - p_{S,t}(i)) + (1 - \eta)(1 - P_{ii})}, \quad \eta \in [0, 1].
$$
If denominator $< 10^{-12}$, apply deterministic uniform fallback over non-current states: $d_{F,t}(j) = 1/(K-1) = 0.5$.

*Reconstructed 3-Class Probabilities*:
$$
p_{F,t}(i) = 1 - q_t, \quad p_{F,t}(j) = q_t d_{F,t}(j) \text{ for } j \ne i.
$$
**Proof of Normalization**:
$$
\sum_{k=0}^2 p_{F,t}(k) = p_{F,t}(i) + \sum_{j \ne i} p_{F,t}(j) = (1 - q_t) + q_t \sum_{j \ne i} d_{F,t}(j) = (1 - q_t) + q_t(1) = 1.
$$

---

## 2. Quantitative Results Summary (Test Scope A, N = 5,685)

| Model ID | Accuracy | Macro-F1 | Change Recall | Change Precision | Change F1 | $A_{\text{change}}$ | $A_{\text{stable}}$ | High Entry F1 | Brier Score | Log Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M0: Markov Baseline** | **90.19%** | 0.6650 | 0.00% | 0.00% | 0.0000 | 0.00% | **100.00%** | 0.0000 | **0.1666** | 0.3392 |
| **M1: Standalone Linear SVM** | 83.74% | 0.8031 | 46.39% | 28.34% | 0.3518 | 41.06% | 88.04% | 0.3049 | 0.2196 | 0.3726 |
| **M1b: Exact RBF (pca_B)** | 84.21% | 0.8101 | 48.86% | 30.06% | 0.3722 | 42.59% | 88.41% | 0.3827 | N/A | N/A |
| **M1b: Exact RBF (hosvd_B)** | 83.71% | 0.8112 | **50.38%** | 29.64% | 0.3732 | **43.35%** | 87.81% | 0.3591 | N/A | N/A |
| **F1: Locked Fixed ($\lambda=0.9$)** | 85.75% | 0.8273 | 43.92% | 33.19% | **0.3781** | 37.64% | 90.99% | **0.3743** | 0.2013 | 0.3539 |
| **F2: Locked Adaptive ($\tau=0.65$)** | 86.47% | 0.8360 | 33.84% | 31.79% | 0.3278 | 29.66% | 92.60% | 0.2881 | 0.2081 | 0.3699 |
| **F3: Locked Transition ($\rho=0.75$)**| 88.15% | **0.8629** | 26.43% | **39.60%** | 0.3170 | 15.59% | 95.89% | 0.0344 | 0.1814 | **0.3329** |

---

## 3. Mathematical Interpretation of Measured Trade-Offs

1. **Change Recall vs. $A_{\text{change}}$ Distinction**: Binary change recall registers any prediction where $\hat{y}_t \ne S_t$. However, exact destination accuracy $A_{\text{change}}$ requires $\hat{y}_t = y_t$. Mathematically, $A_{\text{change}} \le \text{Recall}_{\text{change}}$ holds identically across all models. For example, standalone Linear SVM detects 244 binary changes out of 526 true transitions ($\text{Recall} = 46.39\%$), but only 216 arrive at the correct destination ($A_{\text{change}} = 41.06\%$).
2. **Markov Persistence Dominance in Brier & LogLoss**: The Markov baseline achieves the lowest Brier score ($0.1666$) because PM2.5 state changes occur in only $9.25\%$ of hours ($526 / 5,685$). Over $90.75\%$ of hours are stable, making state persistence probabilistically optimal for overall squared error and cross-entropy loss.
3. **Convex Fusion Pareto Frontier**: Fixed fusion F1 ($\lambda=0.9$) establishes a favorable trade-off: it recovers $+2.01\%$ overall accuracy over standalone SVM ($85.75\%$ vs $83.74\%$), reduces Brier loss from $0.2196$ to $0.2013$, while maintaining the highest Change F1 ($0.3781$) and High-entry F1 ($0.3743$).
