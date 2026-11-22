# Independent mathematical verification

All 24 PM2.5 lags are reconstructed from raw station data using split-local forward fill and the saved training medians. Maximum errors and state matches are in physical_pm25_reconstruction.csv. Independently calculated feature formulas agree with exported features. Explicit sigmoid evaluation matches stored logistic probabilities; separate probability reconstruction agrees with p0/p1/p2. Original saved SVM and F3 probabilities are reproduced by a separate adapter.

Metrics are calculated from scalar counts in metrics_independent.py. Both any-change detection and exact destination are checked; the weighted accuracy decomposition holds. Brier, clipped-renormalized natural log loss and ten-bin confidence ECE agree. Independent paired calendar-block bootstrap agrees with all reported intervals. See the individual reconciliation CSVs for tolerances.
