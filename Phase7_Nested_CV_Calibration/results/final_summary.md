# Phase 7 Final Summary - Nested Cross-Validation Calibration

## Performance Comparison

| Model | Accuracy | Macro-F1 | Change-F1 | J-Score | Brier | NLL | ECE |
|-------|----------|----------|-----------|---------|-------|-----|-----|
| M0_Markov_Baseline | 0.904196 | 0.886177 | 0.000000 | 0.443089 | - | - | 0.017511 |
| M1_Standalone_Linear_Best | 0.837413 | 0.803124 | 0.351838 | 0.577481 | 0.219519 | 0.372767 | 0.015104 |
| F1_Fixed | 0.857517 | 0.827300 | 0.378069 | 0.602684 | - | - | 0.035144 |
| F2_Adaptive | 0.864685 | 0.835978 | 0.327808 | 0.581893 | - | - | 0.032016 |
| RBF_Nested_Calibration | 0.821678 | 0.786352 | 0.359178 | 0.572765 | 0.249292 | 0.503344 | 0.073762 |

## Calibration Metrics (RBF Nested CV)

- **Expected Calibration Error**: 0.0737624
- **Brier Score**: 0.24929157
- **Negative Log-Likelihood**: 0.50334417

## Cost-Sensitive Weight Optimization

| Transition | Stable | High-Entry | J-Score |
|------------|--------|------------|---------|
| 2 | 1.0 | 5 | 0.5745 |
| 2 | 1.0 | 10 | 0.5657 |
| 5 | 1.0 | 5 | 0.5901 |
| 5 | 1.0 | 10 | 0.5795 |
| 10 | 1.0 | 5 | 0.5836 |

**Best Configuration**: transition=5, stable=1.0, high_entry=5
**Best J-Score**: 0.59010359

## Key Findings

1. **Best Overall Model**: F1_Fixed with J=0.602684
2. **Best Calibrated RBF**: JIT=0.572765 with proper nested CV isolation
3. **Calibration Quality**: RBF ECE=0.0738 indicates reasonable calibration
4. **Nested CV Isolation**: Achieved - validation folds never used for training

## Files Generated

- `phase7_summary.json` - Structured results
- `phase7_report.md` - Detailed markdown report  
- `phase7_comparison.csv` - Model comparison table
- `phase7_independent_verification.md` - Complete verification report
- `cost_sensitive_weights.csv` - Weight grid search results
- `rbf_nested_calibration/eces_bins.csv` - ECE binning analysis
- `rbf_nested_calibration/fold_details.csv` - Fold specifications