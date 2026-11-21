# Phase 5 Research Report (post-audit remediation)

*Generated 2026-11-15T14:21:37+00:00 by `source/s11_report_and_audit.py` from the result files in this package. Every number below is read from `results/`, `manifests/` or the prediction CSVs; none is typed in. Independent supervisor review of this remediated package is **pending** — nothing here should be read as audited or approved.*

**Status vocabulary.** *Implemented* = code exists. *Executed* = a log/result file in this package shows it ran. *Independently verified* = reproduced by someone other than the implementer; **nothing in this package has that status yet**.

## 1. Scope of this document
Phase 5 compares linear SVMs, exact RBF kernel SVCs and random-Fourier-feature (RFF) SVMs on PCA and HoSVD representations (Phase 4, immutable) for next-hour PM2.5 state at Aotizhongxin. Targets are **research-defined** (Low < 35, Moderate 35–75, High ≥ 75 µg/m³); 75 is not an official AQI or public-health alert category. All evaluation uses the original class distribution. Scope A = rows with an observed current state (validation n=4,197, test n=5,685); Scope B = the full split (validation n=4,239, test n=5,720). Transition and High-entry metrics exist only on Scope A.

Three result families are kept strictly apart:
* **Original lock** — `manifests/LOCKED_SELECTION.json`, written by `s06` before it opened test features; preserved unchanged.
* **Post-audit revision** — RFF models chosen by a new pre-declared validation-only rule (§7.3). Written *after* the original test results had been seen, so it is **not** a pre-test lock.
* **Stage B addendum** — exact RBF SVCs, run during remediation, selected on validation only and evaluated on test once, after the original lock. Also **not** part of the original lock.

## 2. Data integrity and Phase 3 reference figures

The integrity gate (`s01`) recorded **210 passed / 0 failed / 0 warnings** (`results/integrity_gate.json`, run at 2026-11-15T07:03:53Z). Phase 2–4 files were opened read-only. A fresh re-run of the gate in the remediation environment (2026-11-15T14:20:23Z) gave 210 passed / 0 failed, and all 39 input files common to both runs have identical SHA-256 (0 mismatches); `results/input_checksum_recheck.json`.

Phase 3 reference figures recomputed from the supplied CSVs:

| Quantity | Recomputed from CSV | Spec reference |
|:---|---:|---:|
| Scope A n (validation / test) | 4197 / 5685 | 4,197 / 5,685 |
| Actual transitions (validation / test) | 488 / 526 | 488 / 526 |
| Entries into High (validation / test) | 122 / 154 | 122 / 154 |
| Markov test Scope A accuracy | 0.9075 | ≈ 0.907476 |
| Markov test Scope B accuracy | 0.9042 | ≈ 0.904196 |
| Markov hard-detected changes (test) | 0 | 0 |
| Markov test Scope A Brier / log-loss (unscaled Brier, clipped log-loss) | 0.1666 / 0.3392 | ≈ 0.1666 / ≈ 0.3392 |

## 3. Objectives and conventions (full derivations: `reports/math_chapter.md`)
* **Classical SVM (QP verification only):** hinge loss, unregularised bias, dual 0 ≤ α ≤ C, Σαᵢyᵢ = 0.
* **Linear models (Stage A, and the linear layer of Stage C):** `LinearSVC` (liblinear), **L2-regularised squared-hinge** loss, one-vs-rest, `dual=False`. liblinear appends a constant feature (`intercept_scaling=1`), so the intercept is **regularised**, unlike the classical bias. Its optimum and dual variables are not those of the hinge-loss QP.
* **Exact RBF (Stage B):** `SVC(kernel='rbf')`, libsvm, native one-vs-one, hinge loss, unregularised bias, K = exp(−γ‖x−z‖²) with γ = a/d, d = 192.
* **RFF (Stage C):** ω ~ N(0, 2γI), b ~ U(0, 2π), z(x) = √(2/D)·cos(Wᵀx + b), then `LinearSVC` as above; a finite-dimensional approximation, **not** the same optimiser as exact `SVC`.
* SVM scores are not probabilities; calibrated probabilities come only from §10.

## 4. Mathematical checks that were executed
* Dual QP on synthetic sets: 19/19 internal checks passed; primal–dual gaps 3.47e-08 (separable) and 6.07e-09 (non-separable). These sets are **synthetic**; the spec's additional 60–150-sample *training-only* QP subset was not run unless `results/qp_real_subset.json` exists.
* RBFSampler vs manual sampler (`rff_convention_test.json`): MAE 0.0112 (manual) vs 0.0113 (sklearn), reproducible = True. The test-suite additionally checks the manual √(2/D)·cos(Wᵀx+b) construction equals `RBFSampler` to 1e-12.

* Classical hinge-loss dual QP on a **training-only real subset** (n = 100: 50 Low / 50 High, PCA-B, frozen scaler; `results/qp_real_subset.json`), compared with a matching `SVC(kernel='linear')`: all checks passed = True. d192_C1: SVs 71 (71 free / 0 bound), primal–dual gap 1.7e-05, KKT stationarity 0.0e+00, max complementary-slackness residual 2.0e-06, |b_QP − b_SVC| 7.4e-08; d5_C1: SVs 48 (4 free / 44 bound), primal–dual gap 1.5e-06, KKT stationarity 0.0e+00, max complementary-slackness residual 7.2e-07, |b_QP − b_SVC| 2.8e-07; d5_C10: SVs 45 (6 free / 39 bound), primal–dual gap 4.9e-06, KKT stationarity 0.0e+00, max complementary-slackness residual 1.4e-06, |b_QP − b_SVC| 7.6e-06. Tolerances: bound 1e-8, equality 1e-6, KKT/gap 1e-3. This is the classical hinge-loss problem, not Stage A's squared-hinge LinearSVC.

Kernel approximation on training subsets (γ = 1/192, 5 seeds, `rff_kernel_approx.csv`):

| m | D | ‖K−K̂‖_F/‖K‖_F mean | sd | mean |entry error| mean | sd |
|:---|---:|---:|---:|---:|---:|
| 500 | 256 | 0.3087 | 0.0681 | 0.0495 | 0.0009 |
| 500 | 1024 | 0.1519 | 0.0352 | 0.0243 | 0.0006 |
| 500 | 2048 | 0.1098 | 0.0231 | 0.0177 | 0.0006 |
| 1000 | 256 | 0.2629 | 0.0586 | 0.0490 | 0.0011 |
| 1000 | 1024 | 0.1292 | 0.0315 | 0.0240 | 0.0007 |
| 1000 | 2048 | 0.0940 | 0.0210 | 0.0176 | 0.0012 |

At m = 1000 the mean relative Frobenius error falls from 0.2629 (D=256) to 0.0940 (D=2048), a factor 2.80 for an 8-fold increase in D (a pure D^(−1/2) law would give 2.83). Seed-to-seed spread is non-trivial, and neither individual seeds nor downstream classification scores need be monotone in D.

## 5. Stage A — 40 full-training LinearSVC fits (validation)
All 40 fits are in `results/stage_a_all_candidates.csv`. Best C per representation by validation Scope B macro-F1 (n=4,239):

| Representation | C | Macro-F1 | Accuracy | Scope A transition F1 | Fit time (s) |
|:---|---:|---:|---:|---:|---:|
| pca_A | 0.0100 | 0.6996 | 0.7686 | 0.3438 | 0.0874 |
| pca_B | 0.1000 | 0.7628 | 0.8051 | 0.3872 | 0.5279 |
| pca_C | 1.0000 | 0.7734 | 0.8061 | 0.3824 | 1.9832 |
| hosvd_A | 100.0000 | 0.5942 | 0.7115 | 0.3213 | 0.1713 |
| hosvd_B | 0.0100 | 0.7108 | 0.7707 | 0.3393 | 1.2335 |
| hosvd_C | 0.1000 | 0.7443 | 0.7874 | 0.3604 | 6.0494 |
| hosvd_B_alt1 | 1.0000 | 0.7346 | 0.7766 | 0.3384 | 1.5410 |
| hosvd_B_alt2 | 1.0000 | 0.7219 | 0.7721 | 0.3556 | 0.9823 |

PCA vs HoSVD with the same solver family and C grid (validation, one fit per cell, **no uncertainty interval**):

| k | PCA macro-F1 | HoSVD macro-F1 | PCA − HoSVD |
|:---|---:|---:|---:|
| 48 | 0.6996 | 0.5942 | 0.1053 |
| 192 | 0.7628 | 0.7108 | 0.0521 |
| 576 | 0.7734 | 0.7443 | 0.0291 |

PCA is higher at every k in this validation comparison. This is a statement about these linear fits only; §6 shows the gap is not a property of the representations in general.

HoSVD rank allocations at d = 192 (validation, linear):  hosvd_B = 0.7108;  hosvd_B_alt1 = 0.7346;  hosvd_B_alt2 = 0.7219. Spread 0.0238; single fits, so ranks are suggestive only.

**Selection-rule audit.** The documented tie rule (§12 of the spec: within 0.005 of the best macro-F1, prefer Scope A transition F1, then recall, balanced accuracy, cost) was *not* what the original code applied (it used plain arg-max). Applying the rule to the stored validation results would change the chosen C for 5 of 8 representations (hosvd_A, hosvd_B, hosvd_B_alt2, pca_A, pca_B). `results/stage_a_tie_rule_audit.csv` lists both. The original lock is **not** altered; this is reported as a finding.

## 6. Stage B — exact RBF SVC (native one-vs-one)

Grid executed: **36 of 36** cells completed (2 representations × 2 training sizes × γ-scale a ∈ {0.1, 1, 10} × C ∈ {0.1, 1, 10}); statuses: {'done': 36}. Source: `results/stage_b_all_candidates.csv`, logs `logs/s04_stage_b.log`. Scalers: the frozen Stage A scalers were loaded with the corrected key handling (`mean`/`scale`); no scaler was fitted in this stage.

Training subsets (deterministic systematic temporal blocks of 100 consecutive rows spread over the whole training period; identical rows for both representations):

| fit_n | Low | Moderate | High | share Low | share Mod | share High | months covered (of 34) | TV distance vs full train |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4000 | 1368 | 835 | 1797 | 0.3420 | 0.2087 | 0.4492 | 34 | 0.0332 |
| 8000 | 2788 | 1783 | 3429 | 0.3485 | 0.2229 | 0.4286 | 34 | 0.0220 |
Full-training class shares: Low 0.3265, Moderate 0.2419, High 0.4316.

Best cell per (representation, fit_n) by validation Scope B macro-F1:

| Repr. | fit_n | a (γ=a/192) | C | Macro-F1 | Accuracy | Trans. F1 (Scope A) | # SVs | fit s | peak RSS MB | model bytes |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hosvd_B | 4000 | 0.1 | 10 | 0.7084 | 0.7282 | 0.3072 | 2034 | 0.76 | 239 | 3,166,110 |
| hosvd_B | 8000 | 0.1 | 10 | 0.7604 | 0.7870 | 0.3361 | 4033 | 2.61 | 285 | 6,276,554 |
| pca_B | 4000 | 0.1 | 10 | 0.7414 | 0.7596 | 0.3433 | 2005 | 0.76 | 239 | 3,120,986 |
| pca_B | 8000 | 0.1 | 10 | 0.7404 | 0.7799 | 0.3610 | 3900 | 2.26 | 284 | 6,069,606 |

**Grid-boundary caution:** the best cell sits on an edge of the prescribed grid (smallest γ-scale 0.1 and/or largest C 10) for 4 of 4 (representation, fit_n) pairs; better hyper-parameters may lie outside the grid. The grid was kept as specified and not extended.

Validation-only Stage B selection (rule: macro-F1, 0.005 tie band): pca_B: B_pca_B_n8000_a0.1_C10.0 (macro-F1 0.7404); hosvd_B: B_hosvd_B_n8000_a0.1_C10.0 (macro-F1 0.7604).

Memory/time measurement: fits ran in forked child processes with an enforced wall-time limit; peak RSS is `getrusage` in the child (includes copy-on-write pages shared with the parent, so it is an upper-ish bound). The RAM guard is an estimate, not an OS cap. No dense N×N Gram matrix is built (the 24,108² float64 reference of ≈4.33 GiB is only a reference figure).

**OvO vs OvR ablation (fit_n = 4000, same rows, scaler, γ, C).** The audited script only toggled `decision_function_shape`, which cannot alter `SVC.predict`; the corrected ablation trains a genuine one-vs-rest model:

| Repr. | Variant | Macro-F1 | Accuracy | Trans. F1 | total SVs | fit s |
|:---|---:|---:|---:|---:|---:|---:|
| pca_B | OvO_native_libsvm | 0.7414 | 0.7596 | 0.3433 | 2005 | 0.7013 |
| pca_B | OvO_native_with_dfs_ovo_toggle (old 'ablation') | 0.7414 | 0.7596 | 0.3433 | 2005 | 0.6062 |
| pca_B | OvR_trained_OneVsRestClassifier | 0.6548 | 0.7306 | 0.3268 | 4375 | 1.4170 |
| hosvd_B | OvO_native_libsvm | 0.7084 | 0.7282 | 0.3072 | 2034 | 0.6587 |
| hosvd_B | OvO_native_with_dfs_ovo_toggle (old 'ablation') | 0.7084 | 0.7282 | 0.3072 | 2034 | 0.6917 |
| hosvd_B | OvR_trained_OneVsRestClassifier | 0.6179 | 0.7148 | 0.3263 | 4488 | 1.4162 |

Check that the old toggle changes nothing: pca_B: identical=True; hosvd_B: identical=True. Residual differences between OvO and OvR here: each OvR binary SVC sees all rows, OvO sees pairs; the aggregation rule and the number of support vectors differ; the comparison is one hyper-parameter setting (the OvO winner's), which may favour OvO.

**Optional full-training exact fit (24,108 rows; validation only at this stage).** Hyper-parameters were those of the fit_n = 8000 validation winner (not re-tuned at full size):

| Repr. | fit_n | a | C | Macro-F1 | Accuracy | Trans. F1 | High-entry recall | # SVs | fit s | peak RSS MB | model bytes |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pca_B | 24108 | 0.1 | 10 | 0.8098 | 0.8306 | 0.4031 | 0.3934 | 11306 | 19.7 | 645 | 17,593,351 |
| hosvd_B | 24108 | 0.1 | 10 | 0.8099 | 0.8250 | 0.3870 | 0.4098 | 11746 | 23.2 | 670 | 18,277,991 |

These fits completed well inside the 3 GB / 10 min guidance on the hardware used; that is a measurement on that machine, not a guarantee elsewhere. Exact-SVC models trained on 4,000/8,000 rows are **not** an apples-to-apples comparison with 24,108-row RFF/linear models; the full-train rows are labelled fit_n = 24,108.

## 7. Stage C — RFF + linear SVM

Coarse (a, C) tuning at D = 1024, seed 42 (validation): best pca_B: a=0.1, C=10 (macro-F1 0.7262); hosvd_B: a=0.1, C=10 (macro-F1 0.7170). Both winners are at the grid edge (a = 0.1, C = 10), so the tuned region may not contain the optimum.

### 7.1 Seed variability (validation, Scope B macro-F1, n=4,239; all five predeclared seeds 42–46 retained)

| Repr. | D | mean | sd | min | median | max |
|:---|---:|---:|---:|---:|---:|---:|
| hosvd_B | 256 | 0.6081 | 0.0201 | 0.5898 | 0.6019 | 0.6351 |
| hosvd_B | 1024 | 0.7231 | 0.0109 | 0.7097 | 0.7213 | 0.7371 |
| hosvd_B | 2048 | 0.7351 | 0.0085 | 0.7221 | 0.7396 | 0.7421 |
| pca_B | 256 | 0.6415 | 0.0066 | 0.6336 | 0.6436 | 0.6497 |
| pca_B | 1024 | 0.7211 | 0.0085 | 0.7122 | 0.7219 | 0.7320 |
| pca_B | 2048 | 0.7408 | 0.0043 | 0.7350 | 0.7416 | 0.7465 |

### 7.2 What the original lock did

The original lock took the arg-max over single validation rows: rff_pca_B: D=2048, seed=43 (rank 1 of 5, best single seed = True); rff_hosvd_B: D=2048, seed=44 (rank 1 of 5, best single seed = True). That is best-seed selection, contrary to the predetermined-seed protocol, and the two representations used different random frequencies. The original lock and its test numbers are preserved and shown below as such.

### 7.3 Revised validation-only rule (written during remediation, before any revised model was fit or scored)

Rule: median-seed common pair (see select_rff_policy docstring). Applied to the stored 30 runs: D* = 2048; ranking of seeds by pair-mean validation macro-F1 (best→worst) = [43, 45, 44, 42, 46]; selected seed* = 44 (the median-ranked seed). Both representations use (D*, seed*) so they share identical random frequencies and phases. The rule never picks the best or worst seed *of the pair-mean ranking*; for pca_B alone, seed 44 ranks 4 of 5 on validation macro-F1; for hosvd_B alone, seed 44 ranks 1 of 5 on validation macro-F1, so a single representation's best seed can still coincide with the common seed. This revision was written after the original test results were known, so it is a *post-audit analysis*, not a pre-test lock. No test metric entered the rule (`test_metrics_used_for_selection = false`).

## 8. Selection manifests

Original lock (`manifests/LOCKED_SELECTION.json`, 2026-11-15T07:37:40.385460+00:00):

| Model id | Specification | Validation Scope B macro-F1 |
|:---|---:|---:|
| linear_pca_B | LinearSVC_OvR, pca_B, C=0.1 | 0.7628 |
| linear_hosvd_B | LinearSVC_OvR, hosvd_B, C=0.01 | 0.7108 |
| best_linear_overall | LinearSVC_OvR, pca_C, C=1 | 0.7734 |
| rff_pca_B | RFF_LinearSVC_OvR, pca_B, C=10, D=2048, seed=43 | 0.7465 |
| rff_hosvd_B | RFF_LinearSVC_OvR, hosvd_B, C=10, D=2048, seed=44 | 0.7421 |

Revised selection: `manifests/REVISED_SELECTION_post_audit.json`. Stage B: `manifests/STAGE_B_SELECTION_validation_only.json`. All three are validation-only.

## 9. Test results (each model evaluated once on test; no test-driven selection)
Scope B n = 5,720; Scope A n = 5,685; transition and High-entry rows use Scope A only. Zero-denominator convention shared by all models: undefined precision/recall/F1 = 0.

### 9.1 Phase 3 baselines (recomputed from the supplied CSVs)

| Model | B accuracy | B macro-F1 | B balanced acc. | A accuracy | A macro-F1 |
|:---|---:|---:|---:|---:|---:|
| Phase 3 majority | 0.4154 | 0.1957 | 0.3333 | 0.4157 | 0.1957 |
| Phase 3 persistence | 0.9042 | 0.8862 | 0.8858 | 0.9075 | 0.8895 |
| Phase 3 Markov | 0.9042 | 0.8862 | 0.8858 | 0.9075 | 0.8895 |

| Model | actual changes | predicted changes | TP | FP | FN | precision | recall | F1 | dest. acc. | High TP | FP | FN | High prec. | High rec. | High F1 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Phase 3 majority | 526 | 3325 | 375 | 2950 | 151 | 0.1128 | 0.7129 | 0.1948 | 0.2928 | 154 | 3171 | 0 | 0.0463 | 1.0000 | 0.0885 |
| Phase 3 persistence | 526 | 0 | 0 | 0 | 526 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 154 | 0.0000 | 0.0000 | 0.0000 |
| Phase 3 Markov | 526 | 0 | 0 | 0 | 526 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 154 | 0.0000 | 0.0000 | 0.0000 |

### 9.2 Original lock
Values come from the saved prediction CSVs; they reconcile with `final_evaluation_scopeAB.csv` to within 3.9e-16.


| Model | B accuracy | B macro-F1 | B balanced acc. | A accuracy | A macro-F1 |
|:---|---:|---:|---:|---:|---:|
| Linear PCA-B (orig. lock) | 0.8262 | 0.7614 | 0.7591 | 0.8266 | 0.7617 |
| Linear HoSVD-B (orig. lock) | 0.8059 | 0.7327 | 0.7332 | 0.8058 | 0.7323 |
| Linear PCA-C (orig. lock, best linear) | 0.8337 | 0.7907 | 0.7882 | 0.8336 | 0.7904 |
| RFF PCA-B (orig. lock) | 0.8063 | 0.7523 | 0.7494 | 0.8065 | 0.7526 |
| RFF HoSVD-B (orig. lock) | 0.7946 | 0.7397 | 0.7378 | 0.7945 | 0.7395 |

| Model | actual changes | predicted changes | TP | FP | FN | precision | recall | F1 | dest. acc. | High TP | FP | FN | High prec. | High rec. | High F1 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Linear PCA-B (orig. lock) | 526 | 950 | 268 | 682 | 258 | 0.2821 | 0.5095 | 0.3631 | 0.4221 | 97 | 268 | 57 | 0.2658 | 0.6299 | 0.3738 |
| Linear HoSVD-B (orig. lock) | 526 | 1083 | 286 | 797 | 240 | 0.2641 | 0.5437 | 0.3555 | 0.4163 | 107 | 404 | 47 | 0.2094 | 0.6948 | 0.3218 |
| Linear PCA-C (orig. lock, best linear) | 526 | 916 | 266 | 650 | 260 | 0.2904 | 0.5057 | 0.3689 | 0.4373 | 71 | 150 | 83 | 0.3213 | 0.4610 | 0.3787 |
| RFF PCA-B (orig. lock) | 526 | 1061 | 273 | 788 | 253 | 0.2573 | 0.5190 | 0.3440 | 0.4068 | 90 | 304 | 64 | 0.2284 | 0.5844 | 0.3285 |
| RFF HoSVD-B (orig. lock) | 526 | 1133 | 278 | 855 | 248 | 0.2454 | 0.5285 | 0.3351 | 0.4049 | 86 | 348 | 68 | 0.1982 | 0.5584 | 0.2925 |

### 9.3 Post-audit revision (RFF with the median-seed common-pair rule)
**Not a pre-test lock** (see §7.3).


| Model | B accuracy | B macro-F1 | B balanced acc. | A accuracy | A macro-F1 |
|:---|---:|---:|---:|---:|---:|
| RFF PCA-B (revised, common seed) | 0.7993 | 0.7405 | 0.7387 | 0.7993 | 0.7402 |
| RFF HoSVD-B (revised, common seed) | 0.7944 | 0.7395 | 0.7376 | 0.7944 | 0.7393 |

| Model | actual changes | predicted changes | TP | FP | FN | precision | recall | F1 | dest. acc. | High TP | FP | FN | High prec. | High rec. | High F1 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RFF PCA-B (revised, common seed) | 526 | 1116 | 278 | 838 | 248 | 0.2491 | 0.5285 | 0.3386 | 0.4240 | 98 | 317 | 56 | 0.2361 | 0.6364 | 0.3445 |
| RFF HoSVD-B (revised, common seed) | 526 | 1134 | 278 | 856 | 248 | 0.2451 | 0.5285 | 0.3349 | 0.4049 | 86 | 347 | 68 | 0.1986 | 0.5584 | 0.2930 |

### 9.4 Stage B addendum (exact RBF SVC)
Subset rows (`stageB_*`) are trained on the validation-selected fit_n; `stageB_full_*` on all 24,108 rows with the fit_n = 8000 hyper-parameters. Selected on validation only; evaluated on test after the original lock, so **not part of it**. Probabilities were not produced for exact SVC (no calibrator was fitted); those cells are missing by design.


| Model | B accuracy | B macro-F1 | B balanced acc. | A accuracy | A macro-F1 |
|:---|---:|---:|---:|---:|---:|
| Exact RBF PCA-B (subset) | 0.7913 | 0.7373 | 0.7360 | 0.7912 | 0.7370 |
| Exact RBF HoSVD-B (subset) | 0.8030 | 0.7595 | 0.7577 | 0.8026 | 0.7589 |
| Exact RBF PCA-B (full train) | 0.8421 | 0.8101 | 0.8097 | 0.8417 | 0.8094 |
| Exact RBF HoSVD-B (full train) | 0.8371 | 0.8112 | 0.8147 | 0.8369 | 0.8110 |

| Model | actual changes | predicted changes | TP | FP | FN | precision | recall | F1 | dest. acc. | High TP | FP | FN | High prec. | High rec. | High F1 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Exact RBF PCA-B (subset) | 526 | 1172 | 286 | 886 | 240 | 0.2440 | 0.5437 | 0.3369 | 0.4278 | 96 | 241 | 58 | 0.2849 | 0.6234 | 0.3910 |
| Exact RBF HoSVD-B (subset) | 526 | 1093 | 277 | 816 | 249 | 0.2534 | 0.5266 | 0.3422 | 0.4183 | 89 | 273 | 65 | 0.2459 | 0.5779 | 0.3450 |
| Exact RBF PCA-B (full train) | 526 | 855 | 257 | 598 | 269 | 0.3006 | 0.4886 | 0.3722 | 0.4259 | 71 | 146 | 83 | 0.3272 | 0.4610 | 0.3827 |
| Exact RBF HoSVD-B (full train) | 526 | 894 | 265 | 629 | 261 | 0.2964 | 0.5038 | 0.3732 | 0.4335 | 72 | 175 | 82 | 0.2915 | 0.4675 | 0.3591 |

Observation: the Phase 3 Markov hard predictions predict 0 changes against 526 actual transitions yet have Scope A accuracy 0.9075. Hard accuracy is not a measure of transition responsiveness. Linear PCA-B predicts changes (transition recall 0.5095) with precision 0.2821; Markov's Scope B macro-F1 (0.8862) is higher than every SVM-family model's (best 0.8112). The SVM models use only the 24-hour history; the Markov baseline uses the current state, so they have different information sets.

## 10. Probabilistic evaluation and calibration
* Raw SVM scores are not probabilities. For each final linear/RFF model a multinomial logistic regression (C = 1) maps the three class scores to probabilities.
* **Protocol (train only):** the chronological first 80 % of training rows fit a base classifier; its scores on the final 20 % tail (true labels) fit the mapper. The deployed classifier is then refit on 100 % of training data and the frozen mapper applied. This is a **single forward split, not multi-fold cross-fitting**, and the score scale of the 80 % model is only an approximation of the 100 % model's. No validation/test label or score fits any scaler, classifier, RFF sampler or calibrator.
* **Limitation:** Phase 4 PCA/HoSVD bases were fitted on the whole training period, so even the 80/20 split is not end-to-end forward-chaining at the representation level.
* Class order is [0, 1, 2] = [Low, Moderate, High] for scores, calibrator output and metrics (asserted in code and tests). Raw hard prediction (`hard_pred`) and calibrated argmax (`calibrated_argmax`) are stored separately; **all hard-classification tables use `hard_pred`**.
* Brier = mean Σₖ(pₖ − 1[y=k])² (unscaled; Phase 3 is reproduced under the same convention, §2). Log-loss clips probabilities to [1e-12, 1] and renormalises rows.

| Model (test) | Scope A Brier | Scope A log-loss | Scope B Brier | Scope B log-loss | fraction where calibrated argmax ≠ hard label |
|:---|---:|---:|---:|---:|---:|
| Linear PCA-C (orig. lock, best linear) | 0.2196 | 0.3726 | 0.2195 | 0.3728 | 0.1205 |
| Linear HoSVD-B (orig. lock) | 0.2180 | 0.3990 | 0.2180 | 0.3986 | 0.1098 |
| Linear PCA-B (orig. lock) | 0.2262 | 0.3919 | 0.2262 | 0.3919 | 0.1131 |
| RFF HoSVD-B (revised, common seed) | 0.2303 | 0.4133 | 0.2303 | 0.4132 | 0.1191 |
| RFF PCA-B (revised, common seed) | 0.2539 | 0.4544 | 0.2538 | 0.4541 | 0.1402 |
| RFF HoSVD-B (orig. lock) | 0.2303 | 0.4133 | 0.2303 | 0.4132 | 0.1194 |
| RFF PCA-B (orig. lock) | 0.2510 | 0.4533 | 0.2510 | 0.4532 | 0.1343 |
| Phase 3 majority | 0.6465 | 1.0663 | 0.6466 | 1.0664 | n/a |
| Phase 3 Markov | 0.1666 | 0.3392 | 0.1696 | 0.3438 | n/a |

Phase 3 Markov/majority probabilities come from the supplied CSV columns. Exact-SVC models have no probabilities (missing, not imputed). Reliability diagram for High: `figures/reliability_high_test.png`. Probabilistic scores of different families are comparable only to the extent that each calibrator is adequate; a lower Brier does not by itself establish a better-calibrated model.

## 11. RFF seed variability on the test split (descriptive only)
All five predeclared seeds at D* = 2048 were refit on training data and scored on test. This table shows how much a seed choice can move results; it was **not** used to select anything.

| Repr. | seed | revised rule picks | B macro-F1 | A trans. F1 | A trans. recall | A High-entry recall |
|:---|---:|---:|---:|---:|---:|---:|
| hosvd_B | 42 |  | 0.7413 | 0.3498 | 0.5646 | 0.6234 |
| hosvd_B | 43 |  | 0.7452 | 0.3529 | 0.5437 | 0.6494 |
| hosvd_B | 44 | yes | 0.7395 | 0.3349 | 0.5285 | 0.5584 |
| hosvd_B | 45 |  | 0.7522 | 0.3552 | 0.5513 | 0.6169 |
| hosvd_B | 46 |  | 0.7428 | 0.3534 | 0.5532 | 0.6299 |
| pca_B | 42 |  | 0.7552 | 0.3384 | 0.5038 | 0.5519 |
| pca_B | 43 |  | 0.7520 | 0.3436 | 0.5190 | 0.5844 |
| pca_B | 44 | yes | 0.7405 | 0.3386 | 0.5285 | 0.6364 |
| pca_B | 45 |  | 0.7519 | 0.3398 | 0.5152 | 0.5390 |
| pca_B | 46 |  | 0.7466 | 0.3432 | 0.5266 | 0.6623 |

pca_B: test macro-F1 across seeds ranges 0.7405–0.7552 (sd 0.0058). 
hosvd_B: test macro-F1 across seeds ranges 0.7395–0.7522 (sd 0.0049). 

For scale: the seed-to-seed range of test macro-F1 is 0.0147 (PCA-B) and 0.0127 (HoSVD-B), against a PCA-B − HoSVD-B difference of +0.0287 for the linear models and +0.0010 for the revised RFF pair. RFF seed variability is a separate source of uncertainty from the bootstrap in §12 (which holds the fitted models fixed).

**Refit reproducibility check.** The original lock used seed {oc['rff_hosvd_B']['seed']} for HoSVD-B and the revised rule picks seed {rv['seed_star']}; the revised HoSVD-B model is therefore a refit of the original configuration (same D, γ, C, seed) in a different software environment: 2 of 5720 test hard predictions differ between the original and the refit (likely floating-point/solver or library-version differences; the original environment's package versions were not recorded in the package). For PCA-B the revised seed differs from the original seed, so 602 differing predictions there is expected and is not a reproducibility measure.

## 12. Paired chronological block bootstrap (test)

Resampling unit: **real-time blocks** (calendar 24-hour primary, 168-hour sensitivity) derived from `target_timestamp`, so missing hours do not stretch a block; 242 blocks of 24 h, 1000 resamples, seed 42; the same resampled blocks are used for both models of a pair. Uncertainty covers **test-period sampling only** — not training-set variability, not RFF seed variability — and ignores dependence longer than a block. Intervals are percentile 95 %; no p-values or significance tests are claimed.

**Correction.** The superseded pre-audit bootstrap artifact (239 blocks of 24 consecutive *row-index* rows) recorded, for Linear PCA-B minus HoSVD-B, a macro-F1 difference interval of [+0.0054, +0.0513] and a transition-F1 difference interval of [-0.0170, +0.0316] — which includes zero. The pre-audit report and audit summary instead stated 0.0264 to 0.0601 (macro-F1) and 0.0068 to 0.0664 (transition-F1) and called the PCA advantage statistically robust. Those statements did not match the artifact and are withdrawn. The corrected analysis below uses timestamp-based blocks.

### 24-hour blocks (242 blocks)

| Comparison (a − b) | Metric | Observed diff | 95% low | 95% high | Interval includes 0 |
|:---|---:|---:|---:|---:|---:|
| Linear SVM: PCA-B minus HoSVD-B (original lock) | macro-f1 | 0.0287 | 0.0045 | 0.0513 | no |
| Linear SVM: PCA-B minus HoSVD-B (original lock) | trans-f1 | 0.0076 | -0.0172 | 0.0337 | yes |
| Linear SVM: PCA-B minus HoSVD-B (original lock) | high-f1 | 0.0520 | 0.0108 | 0.0948 | no |
| RFF SVM: PCA-B minus HoSVD-B (revised common seed) | macro-f1 | 0.0010 | -0.0201 | 0.0192 | yes |
| RFF SVM: PCA-B minus HoSVD-B (revised common seed) | trans-f1 | 0.0037 | -0.0196 | 0.0263 | yes |
| RFF SVM: PCA-B minus HoSVD-B (revised common seed) | high-f1 | 0.0514 | 0.0074 | 0.0944 | no |
| PCA-B: linear minus RFF | macro-f1 | 0.0209 | 0.0037 | 0.0399 | no |
| PCA-B: linear minus RFF | trans-f1 | 0.0245 | 0.0032 | 0.0470 | no |
| PCA-B: linear minus RFF | high-f1 | 0.0293 | -0.0068 | 0.0647 | yes |
| HoSVD-B: linear minus RFF | macro-f1 | -0.0068 | -0.0247 | 0.0101 | yes |
| HoSVD-B: linear minus RFF | trans-f1 | 0.0206 | 0.0000 | 0.0420 | no |
| HoSVD-B: linear minus RFF | high-f1 | 0.0288 | -0.0017 | 0.0609 | yes |
| Exact RBF SVC (fit_n subset): PCA-B minus HoSVD-B | macro-f1 | -0.0221 | -0.0445 | -0.0029 | no |
| Exact RBF SVC (fit_n subset): PCA-B minus HoSVD-B | trans-f1 | -0.0053 | -0.0298 | 0.0171 | yes |
| Exact RBF SVC (fit_n subset): PCA-B minus HoSVD-B | high-f1 | 0.0461 | -0.0005 | 0.0923 | yes |
| PCA-B: RFF (full train) minus exact RBF (subset) | macro-f1 | 0.0032 | -0.0142 | 0.0205 | yes |
| PCA-B: RFF (full train) minus exact RBF (subset) | trans-f1 | 0.0017 | -0.0221 | 0.0238 | yes |
| PCA-B: RFF (full train) minus exact RBF (subset) | high-f1 | -0.0466 | -0.0886 | -0.0034 | no |
| Exact RBF SVC (FULL train): PCA-B minus HoSVD-B | macro-f1 | -0.0011 | -0.0173 | 0.0142 | yes |
| Exact RBF SVC (FULL train): PCA-B minus HoSVD-B | trans-f1 | -0.0010 | -0.0286 | 0.0288 | yes |
| Exact RBF SVC (FULL train): PCA-B minus HoSVD-B | high-f1 | 0.0236 | -0.0341 | 0.0813 | yes |
| PCA-B: exact RBF (full train) minus RFF (full train) | macro-f1 | 0.0696 | 0.0525 | 0.0898 | no |
| PCA-B: exact RBF (full train) minus RFF (full train) | trans-f1 | 0.0336 | 0.0073 | 0.0640 | no |
| PCA-B: exact RBF (full train) minus RFF (full train) | high-f1 | 0.0383 | -0.0089 | 0.0868 | yes |
| PCA-B: exact RBF (full train) minus linear | macro-f1 | 0.0487 | 0.0272 | 0.0705 | no |
| PCA-B: exact RBF (full train) minus linear | trans-f1 | 0.0091 | -0.0234 | 0.0417 | yes |
| PCA-B: exact RBF (full train) minus linear | high-f1 | 0.0090 | -0.0297 | 0.0470 | yes |

### 168-hour blocks (35 blocks)

| Comparison (a − b) | Metric | Observed diff | 95% low | 95% high | Interval includes 0 |
|:---|---:|---:|---:|---:|---:|
| Linear SVM: PCA-B minus HoSVD-B (original lock) | macro-f1 | 0.0287 | 0.0039 | 0.0516 | no |
| Linear SVM: PCA-B minus HoSVD-B (original lock) | trans-f1 | 0.0076 | -0.0172 | 0.0315 | yes |
| Linear SVM: PCA-B minus HoSVD-B (original lock) | high-f1 | 0.0520 | 0.0112 | 0.0964 | no |
| RFF SVM: PCA-B minus HoSVD-B (revised common seed) | macro-f1 | 0.0010 | -0.0198 | 0.0205 | yes |
| RFF SVM: PCA-B minus HoSVD-B (revised common seed) | trans-f1 | 0.0037 | -0.0218 | 0.0279 | yes |
| RFF SVM: PCA-B minus HoSVD-B (revised common seed) | high-f1 | 0.0514 | 0.0177 | 0.0829 | no |
| PCA-B: linear minus RFF | macro-f1 | 0.0209 | -0.0009 | 0.0437 | yes |
| PCA-B: linear minus RFF | trans-f1 | 0.0245 | 0.0003 | 0.0521 | no |
| PCA-B: linear minus RFF | high-f1 | 0.0293 | -0.0029 | 0.0618 | yes |
| HoSVD-B: linear minus RFF | macro-f1 | -0.0068 | -0.0283 | 0.0128 | yes |
| HoSVD-B: linear minus RFF | trans-f1 | 0.0206 | -0.0011 | 0.0443 | yes |
| HoSVD-B: linear minus RFF | high-f1 | 0.0288 | -0.0092 | 0.0630 | yes |
| Exact RBF SVC (fit_n subset): PCA-B minus HoSVD-B | macro-f1 | -0.0221 | -0.0462 | -0.0006 | no |
| Exact RBF SVC (fit_n subset): PCA-B minus HoSVD-B | trans-f1 | -0.0053 | -0.0375 | 0.0205 | yes |
| Exact RBF SVC (fit_n subset): PCA-B minus HoSVD-B | high-f1 | 0.0461 | 0.0056 | 0.0819 | no |
| PCA-B: RFF (full train) minus exact RBF (subset) | macro-f1 | 0.0032 | -0.0185 | 0.0266 | yes |
| PCA-B: RFF (full train) minus exact RBF (subset) | trans-f1 | 0.0017 | -0.0200 | 0.0244 | yes |
| PCA-B: RFF (full train) minus exact RBF (subset) | high-f1 | -0.0466 | -0.0862 | -0.0079 | no |
| Exact RBF SVC (FULL train): PCA-B minus HoSVD-B | macro-f1 | -0.0011 | -0.0184 | 0.0149 | yes |
| Exact RBF SVC (FULL train): PCA-B minus HoSVD-B | trans-f1 | -0.0010 | -0.0316 | 0.0318 | yes |
| Exact RBF SVC (FULL train): PCA-B minus HoSVD-B | high-f1 | 0.0236 | -0.0194 | 0.0680 | yes |
| PCA-B: exact RBF (full train) minus RFF (full train) | macro-f1 | 0.0696 | 0.0475 | 0.0921 | no |
| PCA-B: exact RBF (full train) minus RFF (full train) | trans-f1 | 0.0336 | 0.0055 | 0.0633 | no |
| PCA-B: exact RBF (full train) minus RFF (full train) | high-f1 | 0.0383 | -0.0184 | 0.0900 | yes |
| PCA-B: exact RBF (full train) minus linear | macro-f1 | 0.0487 | 0.0233 | 0.0748 | no |
| PCA-B: exact RBF (full train) minus linear | trans-f1 | 0.0091 | -0.0217 | 0.0420 | yes |
| PCA-B: exact RBF (full train) minus linear | high-f1 | 0.0090 | -0.0354 | 0.0513 | yes |

* Linear SVM: PCA-B minus HoSVD-B (original lock): macro-F1 — observed difference +0.0287, 95% interval [+0.0045, +0.0513] — excludes zero under block resampling of the test period (test-sampling uncertainty only). Transition-F1 — observed difference +0.0076, 95% interval [-0.0172, +0.0337] — **includes zero**: this test sample does not separate the two models on this metric.
* RFF SVM: PCA-B minus HoSVD-B (revised common seed): macro-F1 — observed difference +0.0010, 95% interval [-0.0201, +0.0192] — **includes zero**: this test sample does not separate the two models on this metric. Transition-F1 — observed difference +0.0037, 95% interval [-0.0196, +0.0263] — **includes zero**: this test sample does not separate the two models on this metric.
* PCA-B: linear minus RFF: macro-F1 — observed difference +0.0209, 95% interval [+0.0037, +0.0399] — excludes zero under block resampling of the test period (test-sampling uncertainty only). Transition-F1 — observed difference +0.0245, 95% interval [+0.0032, +0.0470] — excludes zero under block resampling of the test period (test-sampling uncertainty only).
* HoSVD-B: linear minus RFF: macro-F1 — observed difference -0.0068, 95% interval [-0.0247, +0.0101] — **includes zero**: this test sample does not separate the two models on this metric. Transition-F1 — observed difference +0.0206, 95% interval [+0.0000, +0.0420] — excludes zero under block resampling of the test period (test-sampling uncertainty only) (a bound lies within 0.0005 of zero, so this is borderline).
* Exact RBF SVC (fit_n subset): PCA-B minus HoSVD-B: macro-F1 — observed difference -0.0221, 95% interval [-0.0445, -0.0029] — excludes zero under block resampling of the test period (test-sampling uncertainty only). Transition-F1 — observed difference -0.0053, 95% interval [-0.0298, +0.0171] — **includes zero**: this test sample does not separate the two models on this metric.
* PCA-B: RFF (full train) minus exact RBF (subset): macro-F1 — observed difference +0.0032, 95% interval [-0.0142, +0.0205] — **includes zero**: this test sample does not separate the two models on this metric. Transition-F1 — observed difference +0.0017, 95% interval [-0.0221, +0.0238] — **includes zero**: this test sample does not separate the two models on this metric.
* Exact RBF SVC (FULL train): PCA-B minus HoSVD-B: macro-F1 — observed difference -0.0011, 95% interval [-0.0173, +0.0142] — **includes zero**: this test sample does not separate the two models on this metric. Transition-F1 — observed difference -0.0010, 95% interval [-0.0286, +0.0288] — **includes zero**: this test sample does not separate the two models on this metric.
* PCA-B: exact RBF (full train) minus RFF (full train): macro-F1 — observed difference +0.0696, 95% interval [+0.0525, +0.0898] — excludes zero under block resampling of the test period (test-sampling uncertainty only). Transition-F1 — observed difference +0.0336, 95% interval [+0.0073, +0.0640] — excludes zero under block resampling of the test period (test-sampling uncertainty only).
* PCA-B: exact RBF (full train) minus linear: macro-F1 — observed difference +0.0487, 95% interval [+0.0272, +0.0705] — excludes zero under block resampling of the test period (test-sampling uncertainty only). Transition-F1 — observed difference +0.0091, 95% interval [-0.0234, +0.0417] — **includes zero**: this test sample does not separate the two models on this metric.

Two different windows (24 h vs 168 h) generally widen the intervals; any conclusion that changes between them is fragile.

## 13. Supplement: transition-weighted training (validation only; not part of winner selection)
Sample weight λ on training rows whose next state differs from the current state (Scope A train rows), 1 otherwise; the current state is used only to define weights — this is not Markov fusion. Evaluated on the natural validation population; single runs at the validation-selected C of each representation, no uncertainty estimate, no calibration.

| Repr. | C | λ | B macro-F1 | trans. precision | trans. recall | trans. F1 | dest. acc. | High prec. | High rec. | High F1 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pca_B | 0.1 | 1 | 0.7628 | 0.3127 | 0.5082 | 0.3872 | 0.4426 | 0.2802 | 0.5328 | 0.3672 |
| pca_B | 0.1 | 2 | 0.7603 | 0.3126 | 0.5246 | 0.3917 | 0.4467 | 0.2692 | 0.5164 | 0.3539 |
| pca_B | 0.1 | 5 | 0.7601 | 0.3082 | 0.5533 | 0.3959 | 0.4713 | 0.2619 | 0.5410 | 0.3529 |
| pca_B | 0.1 | 10 | 0.7631 | 0.2989 | 0.5697 | 0.3921 | 0.5000 | 0.2407 | 0.4754 | 0.3196 |
| hosvd_B | 0.01 | 1 | 0.7108 | 0.2607 | 0.4857 | 0.3393 | 0.4037 | 0.1914 | 0.6230 | 0.2929 |
| hosvd_B | 0.01 | 2 | 0.7168 | 0.2664 | 0.5000 | 0.3476 | 0.4139 | 0.1894 | 0.6148 | 0.2896 |
| hosvd_B | 0.01 | 5 | 0.7291 | 0.2662 | 0.5205 | 0.3523 | 0.4426 | 0.1929 | 0.5820 | 0.2898 |
| hosvd_B | 0.01 | 10 | 0.7220 | 0.2455 | 0.5348 | 0.3366 | 0.4549 | 0.1895 | 0.5328 | 0.2796 |

Effect of weighting (validation, one run each): hosvd_B: λ=1→λ=10 transition recall 0.4857→0.5348, transition F1 0.3393→0.3366, High-entry F1 0.2929→0.2796, macro-F1 0.7108→0.7220 (recall monotone in λ: True; F1 monotone: False); pca_B: λ=1→λ=10 transition recall 0.5082→0.5697, transition F1 0.3872→0.3921, High-entry F1 0.3672→0.3196, macro-F1 0.7628→0.7631 (recall monotone in λ: True; F1 monotone: False). No claim of guaranteed improvement or calibration is made.

## 14. Comparative findings (conditional on the numbers above)

* **PCA vs HoSVD under the exact kernel.** At d = 192 the full-train exact RBF SVC reaches validation macro-F1 0.8098 (PCA-B) and 0.8099 (HoSVD-B), a difference of -0.0002, versus +0.0521 for the linear models. The PCA–HoSVD gap observed for the linear models is not visible under the exact kernel at d = 192 (|difference| ≤ the 0.005 practical tie band, a convention and not a test); single validation fits, no interval.
* **Exact vs approximate kernel.** Validation macro-F1 of the best RFF fits (D = 2048) is 0.7465 (best single run), whereas the exact full-train SVC reaches 0.8099; RFF at D ≤ 2048 with a linear squared-hinge solver is a coarser approximation and uses OvR, while exact SVC uses OvO hinge loss, so the gap mixes approximation, loss and multiclass-strategy effects.
* **Reconstruction vs classification.** Phase 4 reconstruction errors (not recomputed here) measure how well a basis reproduces the inputs; discrimination of the next-hour state need not follow reconstruction quality, and the comparisons above should not be read as a ranking of reconstruction.
* **Training-size confound.** Subset exact fits (4,000/8,000 rows) use far fewer rows than the 24,108-row linear/RFF models.

## 15. Resource notes
Stage A/C timings and sizes were measured on the original run machine (its hardware is not recorded in the result CSVs) and are taken from the stored CSVs; Stage B and the revision stage were run here on a 1-vCPU Linux sandbox with ~3 GB RAM (`logs/s04_stage_b.log`, `logs/s09_post_audit_revision.log`). Timings are therefore **not comparable across stages**. Inference-time benchmarking (warm-up/repeats) and projected-basis storage totals were not re-measured in this remediation.

## 16. Limitations and unexecuted items
1. Phase 4 representations were fitted over the whole original training period; calibration and all internal splits inherit that.
2. Calibration is a single 80/20 chronological split, not multi-fold; Stage B models have no probabilities.
3. Stage A selection used arg-max rather than the documented 0.005 tie rule (impact in §5); the original lock was not changed.
4. Grid-edge optima in Stage B and Stage C; grids were not extended.
5. RFF detailed experiments cover only the PCA-B / HoSVD-B pair; the spec's optional validation-selected alternative pairs and dimension controls (d = 48, 576) for RFF and exact SVC were **not** run.
6. The classical-hinge QP checks use SLSQP on n = 100 points; they verify the dual/KKT mathematics, not the large-scale LinearSVC optimum.
7. Bootstrap intervals reflect test-sampling variability only.
8. Original Stage A, Stage C and `s06` runs were **not re-executed** here; they were verified only by reloading saved models and reconciling saved predictions with metric tables (reload reproduces predictions exactly, see tests). The Stage B, revision, table, bootstrap and report stages were executed in this remediation.
9. Nothing here has been independently verified by anyone but the implementer.

## 17. Test suite
Actual result of `python -m pytest tests` (11.3 s): **66 passed, 0 failed, 0 skipped** of 66. Skipped tests (if any) and reasons are in `results/test_summary.json`; full log in `logs/pytest_log.txt`.
