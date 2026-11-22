# Contribution and authentic ablations

| model | validation_J | validation_change_f1 | validation_macro_f1 | test_J | test_change_f1 | test_macro_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| E0 | 0.43653 | 0.00000 | 0.87307 | 0.44475 | 0.00000 | 0.88951 |
| E1 | 0.56906 | 0.35939 | 0.77874 | 0.57709 | 0.35184 | 0.80233 |
| E2 | 0.59585 | 0.34889 | 0.84281 | 0.59157 | 0.31699 | 0.86615 |
| E3 | 0.61511 | 0.44286 | 0.78736 | 0.60574 | 0.40582 | 0.80566 |
| E4 | 0.62021 | 0.44767 | 0.79275 | 0.60759 | 0.40777 | 0.80742 |
| E5 | 0.61660 | 0.44287 | 0.79033 | 0.62242 | 0.42757 | 0.81727 |
| E6 | 0.59510 | 0.42357 | 0.76664 | 0.60725 | 0.41216 | 0.80235 |
| E7 | 0.57977 | 0.43011 | 0.72943 | 0.58144 | 0.40299 | 0.75988 |

E3 threshold distances and current-state indicators improve transition F1 over F3. E4 adds only target trends and provides a small further increase. E5 adds real cross-station summaries and performs better on this test set, but E4 wins the validation selection and remains the selected final model. The test result is an exploratory observation, not permission to relabel E5 as the winner. E6 changes decisions while preserving E5 probabilities; E7 uses a separately validation-selected all-feature logistic model with alpha=1. Each experiment has separate validation/test prediction CSVs.

The contribution is a verified application and controlled trade-off study. No new theorem, new logistic-regression algorithm, or first-ever use of spatial pollution features is claimed. The evidence does not show that every added feature helps in every split. See literature/literature_gap_analysis.md.
