# Final Phase 7 results

The locked choice is **E4**, classifier `temporal_C1.0_balanced`, alpha=0.75, with probability argmax decisions. It was selected using validation J=0.620206. The C grid was 0.1, 1 and 10; each was evaluated with no weighting and balanced weighting. E3/E4/E5 use 6/8/11 features respectively. E6 keeps E5's trained classifier and alpha and selects theta on validation. E7 fixes alpha=1.

| model | accuracy | macro_f1 | change_precision | change_recall | change_f1 | exact_destination | stable_accuracy | high_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | 0.90748 | 0.88951 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 1.00000 | 0.00000 |
| E1 | 0.83694 | 0.80233 | 0.28339 | 0.46388 | 0.35184 | 0.41065 | 0.88040 | 0.30493 |
| E2 | 0.88461 | 0.86615 | 0.39601 | 0.26426 | 0.31699 | 0.15589 | 0.95891 | 0.03436 |
| E4 | 0.83518 | 0.80742 | 0.32096 | 0.55894 | 0.40777 | 0.40114 | 0.87943 | 0.30263 |

E4 detects 294 of 526 changes (55.89%) versus 139 (26.43%) for F3. False alarms increase from 212 to 622. Accuracy changes from 88.46% to 83.52%, and macro-F1 from 0.8661 to 0.8074. Correct destination accuracy increases from 15.59% to 40.11%, while stable-state accuracy declines from 95.89% to 87.94%. This is a transition-sensitivity trade-off, not an overall accuracy improvement.

All 20 figures and six diagrams are generated with data and plotting source. The separate audit independently checks predictions, equations, metrics and bootstrap. Read FINAL_PROJECT_READINESS.md for remaining academic/evidence qualifications. All prior AP files and the legacy sibling precursor module are preserved.
