# Mathematical Foundations: SVM and Random Fourier Features for Air Pollution Classification

> This chapter provides the rigorous mathematical framework underlying the SVM and RFF-based
> binary (and multi-class) classifiers used in the Phase 5 pipeline.

---

## 5.1 Binary Soft-Margin SVM: Primal Formulation

### 5.1.1 Primal Problem

Given a training set $\{(\mathbf{x}_i, y_i)\}_{i=1}^{n}$ with $\mathbf{x}_i \in \mathbb{R}^d$ and $y_i \in \{+1,-1\}$, the **soft-margin SVM primal** is:

$$
\min_{\mathbf{w},\, b,\, \boldsymbol{\xi}} \quad \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^{n} \xi_i
$$

subject to:
$$
y_i\!\left(\mathbf{w}^\top \mathbf{x}_i + b\right) \geq 1 - \xi_i, \qquad \xi_i \geq 0, \quad i = 1, \ldots, n
$$

where:
- $\mathbf{w} \in \mathbb{R}^d$ is the weight vector (normal to the decision hyperplane),
- $b \in \mathbb{R}$ is the bias (intercept),
- $\xi_i \geq 0$ are **slack variables** that allow misclassification,
- $C > 0$ is the **regularisation constant** trading margin width against training error.

**Geometric interpretation.** The decision rule is $\hat{y}(\mathbf{x}) = \operatorname{sign}(\mathbf{w}^\top \mathbf{x} + b)$.  
The functional margin of sample $i$ is $y_i(\mathbf{w}^\top \mathbf{x}_i + b)$.  
Slack $\xi_i = \max(0,\, 1 - y_i(\mathbf{w}^\top \mathbf{x}_i + b))$ measures the degree of constraint violation.

---

### 5.1.2 Hinge-Loss Equivalent (Unconstrained Primal)

Substituting the optimal slack $\xi_i^\star = \max(0,\, 1 - y_i(\mathbf{w}^\top \mathbf{x}_i + b))$ into the primal objective yields the **unconstrained hinge-loss formulation**:

$$
\min_{\mathbf{w},\, b} \quad \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \max\!\left(0,\; 1 - y_i(\mathbf{w}^\top \mathbf{x}_i + b)\right)
$$

Equivalently, dividing by $C$ and letting $\lambda = 1/C$:

$$
\min_{\mathbf{w},\, b} \quad \frac{\lambda}{2}\|\mathbf{w}\|^2 + \frac{1}{n}\sum_{i=1}^{n} \max\!\left(0,\; 1 - y_i(\mathbf{w}^\top \mathbf{x}_i + b)\right)
$$

**Properties of the hinge loss** $\ell_{\text{hinge}}(t) = \max(0, 1-t)$:
- Convex and piecewise linear.
- Zero for correctly classified points with margin $\geq 1$.
- Grows linearly for misclassified or margin-violating points.
- Sub-gradient: $\partial_t \ell_{\text{hinge}}(t) = \begin{cases} -1 & t < 1 \\ 0 & t > 1 \end{cases}$ (undefined at $t=1$).

---

## 5.2 Classical Dual Derivation

### 5.2.1 Lagrangian

Introducing dual variables $\alpha_i \geq 0$ (for the margin constraints) and $\mu_i \geq 0$ (for $\xi_i \geq 0$), the Lagrangian is:

$$
\mathcal{L}(\mathbf{w}, b, \boldsymbol{\xi}, \boldsymbol{\alpha}, \boldsymbol{\mu}) = \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_i \xi_i - \sum_i \alpha_i\!\left[y_i(\mathbf{w}^\top \mathbf{x}_i + b) - 1 + \xi_i\right] - \sum_i \mu_i \xi_i
$$

### 5.2.2 Stationarity Conditions

Setting partial derivatives to zero:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = 0 \;\Rightarrow\; \mathbf{w} = \sum_{i=1}^{n} \alpha_i y_i \mathbf{x}_i \tag{S1}
$$

$$
\frac{\partial \mathcal{L}}{\partial b} = 0 \;\Rightarrow\; \sum_{i=1}^{n} \alpha_i y_i = 0 \tag{S2}
$$

$$
\frac{\partial \mathcal{L}}{\partial \xi_i} = 0 \;\Rightarrow\; C - \alpha_i - \mu_i = 0 \;\Rightarrow\; \mu_i = C - \alpha_i \tag{S3}
$$

Since $\mu_i \geq 0$, we obtain $\alpha_i \leq C$.

### 5.2.3 Dual Problem

Substituting (S1–S3) back into $\mathcal{L}$ and simplifying gives the **Wolfe dual**:

$$
\boxed{
\max_{\boldsymbol{\alpha}} \quad \sum_{i=1}^{n} \alpha_i - \frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n} \alpha_i \alpha_j y_i y_j\, K(\mathbf{x}_i, \mathbf{x}_j)
}
$$

subject to:
$$
0 \leq \alpha_i \leq C, \qquad \sum_{i=1}^{n} \alpha_i y_i = 0
$$

For the **linear kernel** $K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^\top \mathbf{x}_j$:

$$
\max_{\boldsymbol{\alpha}} \quad \sum_{i} \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j\, \mathbf{x}_i^\top \mathbf{x}_j
$$

subject to $0 \leq \alpha_i \leq C$, $\sum_i \alpha_i y_i = 0$.

**Strong duality** holds because the primal is convex and Slater's condition is satisfied (the problem is strictly feasible for any $C > 0$).

---

## 5.3 KKT Conditions and Complementary Slackness

The Karush–Kuhn–Tucker (KKT) optimality conditions at $(\mathbf{w}^\star, b^\star, \boldsymbol{\xi}^\star, \boldsymbol{\alpha}^\star, \boldsymbol{\mu}^\star)$ are:

### 5.3.1 Stationarity
$$
\mathbf{w}^\star = \sum_i \alpha_i^\star y_i \mathbf{x}_i, \qquad \sum_i \alpha_i^\star y_i = 0, \qquad \mu_i^\star = C - \alpha_i^\star
$$

### 5.3.2 Primal Feasibility
$$
y_i(\mathbf{w}^{\star\top} \mathbf{x}_i + b^\star) \geq 1 - \xi_i^\star, \qquad \xi_i^\star \geq 0
$$

### 5.3.3 Dual Feasibility
$$
\alpha_i^\star \geq 0, \qquad \mu_i^\star = C - \alpha_i^\star \geq 0 \;\Rightarrow\; \alpha_i^\star \leq C
$$

### 5.3.4 Complementary Slackness
$$
\alpha_i^\star \!\left[y_i(\mathbf{w}^{\star\top} \mathbf{x}_i + b^\star) - 1 + \xi_i^\star\right] = 0 \tag{CS1}
$$
$$
\mu_i^\star \xi_i^\star = (C - \alpha_i^\star)\,\xi_i^\star = 0 \tag{CS2}
$$

**Classification of training points** by $\alpha_i^\star$:

| $\alpha_i^\star$ | $\xi_i^\star$ | Point type |
|:---:|:---:|:---|
| $0$ | $0$ | Non-support vector; correctly classified with margin $> 1$ |
| $(0, C)$ | $0$ | **Support vector on the margin hyperplane**; $y_i f(\mathbf{x}_i) = 1$ |
| $C$ | $> 0$ | Support vector inside or beyond margin; misclassified if $\xi_i^\star > 1$ |

### 5.3.5 Recovering the Bias $b^\star$

For any free support vector (i.e., $0 < \alpha_i^\star < C$), $\xi_i^\star = 0$ and:
$$
b^\star = y_i - \mathbf{w}^{\star\top} \mathbf{x}_i
$$

In practice, $b^\star$ is computed as the **average** over all free support vectors for numerical stability.

---

## 5.4 Margin Geometry and Support Vectors

### 5.4.1 Margin Width

The two margin hyperplanes are $\mathbf{w}^\top \mathbf{x} + b = +1$ and $\mathbf{w}^\top \mathbf{x} + b = -1$.  
The perpendicular distance between them is:

$$
\text{margin} = \frac{2}{\|\mathbf{w}\|}
$$

**Derivation.** A unit normal to the hyperplane is $\hat{\mathbf{n}} = \mathbf{w}/\|\mathbf{w}\|$.  
The signed distance of a point $\mathbf{x}$ from the decision boundary is $(\mathbf{w}^\top \mathbf{x} + b)/\|\mathbf{w}\|$.  
The gap between $+1$ and $-1$ margin planes is therefore $(1 - (-1))/\|\mathbf{w}\| = 2/\|\mathbf{w}\|$.

### 5.4.2 Maximising the Margin

Minimising $\frac{1}{2}\|\mathbf{w}\|^2$ is equivalent to maximising $2/\|\mathbf{w}\|$, subject to the feasibility constraints. The soft-margin formulation relaxes the hard constraints by allowing slack, controlled by $C$.

### 5.4.3 Support Vectors

Support vectors are the training points $\mathbf{x}_i$ for which $\alpha_i^\star > 0$.  
The decision function depends **only** on the support vectors:
$$
f(\mathbf{x}) = \mathbf{w}^{\star\top}\mathbf{x} + b^\star = \sum_{i:\,\alpha_i^\star > 0} \alpha_i^\star y_i\, K(\mathbf{x}_i, \mathbf{x}) + b^\star
$$

---

## 5.5 LinearSVC: Squared-Hinge Loss Objective (NOT Classical QP Dual)

> [!IMPORTANT]
> `sklearn.svm.LinearSVC` does **not** solve the classical hinge-loss QP dual. It minimises a
> **squared-hinge loss** with **L2 regularisation**, using the **One-vs-Rest (OvR)** scheme for
> multi-class problems.

### 5.5.1 Exact LinearSVC Objective

For a binary problem, `LinearSVC` (with default `loss='squared_hinge'`, `penalty='l2'`) minimises:

$$
\boxed{
\min_{\mathbf{w},\, b} \quad \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \left[\max\!\left(0,\; 1 - y_i(\mathbf{w}^\top \mathbf{x}_i + b)\right)\right]^2
}
$$

Key differences from the classical SVM:

| Property | Classical SVM (hinge, QP dual) | `LinearSVC` (squared-hinge, primal) |
|:---|:---:|:---:|
| Loss | $\max(0, 1-t)$ | $[\max(0, 1-t)]^2$ |
| Dual variables $\alpha_i$ | Bounded: $0 \leq \alpha_i \leq C$ | Dual differs; $\alpha_i$ can be $> C$ |
| Sparsity of $\boldsymbol{\alpha}$ | Sparse (most $\alpha_i = 0$) | Dense |
| Solver | QP (SMO or interior-point) | LIBLINEAR (primal CD / dual CD) |
| Multi-class | OvO or OvR | **OvR only** |
| Bias regularisation | $b$ not regularised | $b$ can be regularised (via `intercept_scaling`) |

### 5.5.2 Multi-Class OvR Extension

For $K$ classes, `LinearSVC` trains $K$ binary classifiers. Classifier $k$ uses:
- Positive labels: $y_i = +1$ for class $k$.
- Negative labels: $y_i = -1$ for all other classes.

Prediction: $\hat{y}(\mathbf{x}) = \arg\max_{k} \left(\mathbf{w}_k^\top \mathbf{x} + b_k\right)$.

### 5.5.3 Why This Matters for Air Pollution Classification

The pipeline uses three classes (Low / Moderate / High PM2.5). The OvR LinearSVC trains three binary classifiers. The squared-hinge loss makes the dual variables **unbounded**, so support vectors (in the classical sense) are not sparse, and the primal-dual gap interpretation differs from the classical SVM theory.

---

## 5.6 RBF Kernel and Kernelised Dual

### 5.6.1 RBF Kernel Definition

$$
K(\mathbf{x}, \mathbf{z}) = \exp\!\left(-\gamma \|\mathbf{x} - \mathbf{z}\|^2\right), \qquad \gamma > 0
$$

**Properties:**
- Stationary (depends only on $\mathbf{x} - \mathbf{z}$) and isotropic.
- Positive semi-definite (by Bochner's theorem — see §5.7).
- As $\gamma \to 0$: $K \to 1$ (constant kernel, underfitting).
- As $\gamma \to \infty$: $K \to \mathbf{1}[\mathbf{x} = \mathbf{z}]$ (overfitting).
- Corresponds to an infinite-dimensional feature map (RKHS).

### 5.6.2 Kernelised Dual

Replacing $\mathbf{x}_i^\top \mathbf{x}_j$ with $K(\mathbf{x}_i, \mathbf{x}_j)$ in the dual:

$$
\max_{\boldsymbol{\alpha}} \quad \sum_{i} \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j\, e^{-\gamma\|\mathbf{x}_i - \mathbf{x}_j\|^2}
$$

subject to $0 \leq \alpha_i \leq C$, $\sum_i \alpha_i y_i = 0$.

The **kernel matrix** (Gram matrix) $\mathbf{K} \in \mathbb{R}^{n \times n}$ has entries $K_{ij} = K(\mathbf{x}_i, \mathbf{x}_j)$.  
The dual objective in matrix form is:

$$
\max_{\boldsymbol{\alpha}} \quad \mathbf{1}^\top \boldsymbol{\alpha} - \frac{1}{2} \boldsymbol{\alpha}^\top \mathbf{H} \boldsymbol{\alpha}
$$

where $H_{ij} = y_i y_j K_{ij}$.

**Prediction** (kernel expansion):
$$
f(\mathbf{x}) = \sum_{i:\,\alpha_i > 0} \alpha_i y_i\, K(\mathbf{x}_i, \mathbf{x}) + b^\star
$$

---

## 5.7 Random Fourier Features (RFF)

### 5.7.1 Bochner's Theorem

A continuous, bounded, stationary kernel $K(\mathbf{x}, \mathbf{z}) = k(\mathbf{x} - \mathbf{z})$ on $\mathbb{R}^d$ is **positive semi-definite** if and only if it is the Fourier transform of a non-negative measure $p(\boldsymbol{\omega})$:

$$
K(\mathbf{x} - \mathbf{z}) = \int_{\mathbb{R}^d} p(\boldsymbol{\omega})\, e^{j\boldsymbol{\omega}^\top(\mathbf{x} - \mathbf{z})}\, d\boldsymbol{\omega} = \mathbb{E}_{\boldsymbol{\omega} \sim p}\!\left[e^{j\boldsymbol{\omega}^\top(\mathbf{x} - \mathbf{z})}\right]
$$

For real-valued kernels, taking the real part:
$$
K(\mathbf{x}, \mathbf{z}) = \mathbb{E}_{\boldsymbol{\omega} \sim p}\!\left[\cos\!\left(\boldsymbol{\omega}^\top(\mathbf{x} - \mathbf{z})\right)\right]
$$

### 5.7.2 Spectral Density of the RBF Kernel

The Fourier transform of $K(\boldsymbol{\delta}) = \exp(-\gamma\|\boldsymbol{\delta}\|^2)$ is a Gaussian:

$$
p(\boldsymbol{\omega}) = \left(\frac{1}{4\pi\gamma}\right)^{d/2} \exp\!\left(-\frac{\|\boldsymbol{\omega}\|^2}{4\gamma}\right) = \mathcal{N}\!\left(\boldsymbol{\omega};\, \mathbf{0},\, 2\gamma\, \mathbf{I}_d\right)
$$

Therefore the frequency samples are drawn as:
$$
\boldsymbol{\omega}_j \sim \mathcal{N}\!\left(\mathbf{0},\, 2\gamma\, \mathbf{I}_d\right), \qquad j = 1, \ldots, D
$$

### 5.7.3 RFF Feature Map

Using the product-to-sum identity $\cos(a)\cos(b) = \frac{1}{2}[\cos(a-b) + \cos(a+b)]$ and introducing random phase offsets $b_j \sim \mathrm{Uniform}(0, 2\pi)$, Rahimi & Recht (2007) define the $D$-dimensional **Random Fourier Feature (RFF) map**:

$$
\boxed{
z_D(\mathbf{x}) = \sqrt{\frac{2}{D}}\,\cos\!\left(\mathbf{W}^\top \mathbf{x} + \mathbf{b}\right)
}
$$

where:
- $\mathbf{W} = [\boldsymbol{\omega}_1, \ldots, \boldsymbol{\omega}_D] \in \mathbb{R}^{d \times D}$, columns i.i.d.\ $\mathcal{N}(\mathbf{0}, 2\gamma\mathbf{I}_d)$,
- $\mathbf{b} = [b_1, \ldots, b_D]^\top \in \mathbb{R}^D$, entries i.i.d.\ $\mathrm{Uniform}(0, 2\pi)$,
- $\cos(\cdot)$ is applied element-wise.

The resulting feature vector $z_D(\mathbf{x}) \in \mathbb{R}^D$.

### 5.7.4 Expectation Derivation: $\mathbb{E}[z_D(\mathbf{x})^\top z_D(\mathbf{z})] = K(\mathbf{x}, \mathbf{z})$

For a single component $j$:

$$
\mathbb{E}\left[\frac{2}{D}\cos(\boldsymbol{\omega}_j^\top \mathbf{x} + b_j)\cos(\boldsymbol{\omega}_j^\top \mathbf{z} + b_j)\right]
$$

Using $\cos(A)\cos(B) = \frac{1}{2}[\cos(A-B) + \cos(A+B)]$:

$$
= \mathbb{E}\left[\frac{1}{D}\left(\cos(\boldsymbol{\omega}_j^\top(\mathbf{x}-\mathbf{z})) + \cos(\boldsymbol{\omega}_j^\top(\mathbf{x}+\mathbf{z}) + 2b_j)\right)\right]
$$

The second term averages to zero because $b_j \sim \mathrm{Uniform}(0, 2\pi)$:

$$
\mathbb{E}_{b_j}\left[\cos(\boldsymbol{\omega}_j^\top(\mathbf{x}+\mathbf{z}) + 2b_j)\right] = 0
$$

The first term gives:

$$
= \frac{1}{D}\,\mathbb{E}_{\boldsymbol{\omega}_j \sim p}\!\left[\cos\!\left(\boldsymbol{\omega}_j^\top(\mathbf{x}-\mathbf{z})\right)\right] = \frac{1}{D}\, K(\mathbf{x}, \mathbf{z})
$$

Summing over all $D$ independent components (by linearity of expectation):

$$
\mathbb{E}\left[z_D(\mathbf{x})^\top z_D(\mathbf{z})\right] = \sum_{j=1}^{D} \frac{1}{D}\, K(\mathbf{x}, \mathbf{z}) = K(\mathbf{x}, \mathbf{z})
$$

Thus $z_D(\mathbf{x})^\top z_D(\mathbf{z})$ is an **unbiased Monte Carlo estimator** of $K(\mathbf{x}, \mathbf{z})$.

### 5.7.5 Monte Carlo Approximation Error

By the **Central Limit Theorem** (CLT) applied to $D$ i.i.d. terms, each bounded in $[-1,1]$ after scaling, the approximation error concentrates:

$$
\sup_{\mathbf{x}, \mathbf{z}} \left|z_D(\mathbf{x})^\top z_D(\mathbf{z}) - K(\mathbf{x}, \mathbf{z})\right| = \mathcal{O}\!\left(D^{-1/2}\right) \quad \text{with high probability}
$$

More precisely, Rahimi & Recht (2007) show that for any compact set $\mathcal{M} \subset \mathbb{R}^d$ with diameter $\rho$, and $\sigma_p^2 = \mathbb{E}[\boldsymbol{\omega}^\top \boldsymbol{\omega}]$:

$$
\Pr\!\left[\sup_{\mathbf{x},\mathbf{z} \in \mathcal{M}} \left|z_D(\mathbf{x})^\top z_D(\mathbf{z}) - K(\mathbf{x},\mathbf{z})\right| \geq \epsilon\right] \leq 2^8 \left(\frac{\sigma_p \rho}{\epsilon}\right)^2 \exp\!\left(-\frac{D\epsilon^2}{8}\right)
$$

**Practical implication:** To achieve kernel approximation error $\leq \epsilon$ with high probability, $D = \mathcal{O}(\epsilon^{-2}\log(1/\delta))$ features suffice. Typical practice uses $D \in [256, 4096]$, reducing the kernel trick's $\mathcal{O}(n^2)$ cost to $\mathcal{O}(nD)$ for training and $\mathcal{O}(D)$ for prediction.

### 5.7.6 RFF Pipeline Summary

```
Input x ∈ ℝᵈ
    ↓  W ~ N(0, 2γ I_d),  b ~ Uniform(0, 2π)      [fit once on training data]
z_D(x) = √(2/D) · cos(Wᵀx + b) ∈ ℝᴰ              [transform]
    ↓
LinearSVC on {z_D(xᵢ), yᵢ}                         [train]
    ↓
Prediction: sign(wᵀ z_D(x) + b_svm)               [inference, O(D)]
```

**Complexity comparison:**

| Method | Train | Predict | Memory |
|:---|:---:|:---:|:---:|
| RBF-SVM (kernel) | $\mathcal{O}(n^2 d + n^3)$ | $\mathcal{O}(n_{\text{sv}} d)$ | $\mathcal{O}(n^2)$ |
| RFF + LinearSVC | $\mathcal{O}(nDd + nD)$ | $\mathcal{O}(D)$ | $\mathcal{O}(nD)$ |

For the air pollution dataset ($n \approx 24{,}000$, $d \sim 192{,}576$), the kernel SVM is computationally infeasible; RFF makes it tractable.

---

## 5.8 Summary of Key Formulae

| Symbol | Definition |
|:---|:---|
| $\mathbf{w}^\star = \sum_i \alpha_i^\star y_i \mathbf{x}_i$ | Primal weight from dual variables |
| $\text{margin} = 2/\|\mathbf{w}^\star\|$ | Geometric margin width |
| $\alpha_i^\star \in (0, C)$ | Free support vector (on margin) |
| $K(\mathbf{x},\mathbf{z}) = e^{-\gamma\|\mathbf{x}-\mathbf{z}\|^2}$ | RBF kernel |
| $\boldsymbol{\omega}_j \sim \mathcal{N}(\mathbf{0}, 2\gamma\mathbf{I}_d)$ | RFF frequency samples |
| $z_D(\mathbf{x}) = \sqrt{2/D}\cos(\mathbf{W}^\top\mathbf{x}+\mathbf{b})$ | RFF feature map |
| $\mathbb{E}[z_D(\mathbf{x})^\top z_D(\mathbf{z})] = K(\mathbf{x},\mathbf{z})$ | Unbiasedness of RFF |
| MC error $= \mathcal{O}(D^{-1/2})$ | RFF approximation rate |

---

*References:*
- *Boser, B., Guyon, I., & Vapnik, V. (1992). A training algorithm for optimal margin classifiers.*
- *Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine Learning, 20(3), 273–297.*
- *Rahimi, A., & Recht, B. (2007). Random features for large-scale kernel machines. NeurIPS.*
- *Scholkopf, B., & Smola, A. (2002). Learning with Kernels. MIT Press.*
- *Fan, R.-E., Chang, K.-W., Hsieh, C.-J., Wang, X.-R., & Lin, C.-J. (2008). LIBLINEAR: A library for large linear classification.*
