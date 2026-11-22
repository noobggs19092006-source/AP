# Phase 6 to Phase 7 comparison audit

Matching Scope A test rows number 5,685, with 526 actual changes. The independent baseline adapter uses the original model/scaler/calibrator, rho=.75, eta=0 and the original Markov matrix. Its probabilities agree. Missing-current Scope B rows deliberately receive the specified calibrated feature-only SVM fallback; the preserved historical F3 source returned uniform probabilities for those rows. This correction does not change the Scope A comparison.

The separate historical nested summary mixes population/decision conventions. Calibrated argmax differs from raw LinearSVC hard decisions; direct tables label them separately. The legacy Phase 7 attempt is preserved and excluded. See the implementation reconciliation report for exact source paths and discrepancies.
