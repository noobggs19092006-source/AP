# Transition-precursor research report

The locked choice is **E4**, classifier `temporal_C1.0_balanced`, alpha=0.75, with probability argmax decisions. It was selected using validation J=0.620206. The C grid was 0.1, 1 and 10; each was evaluated with no weighting and balanced weighting. E3/E4/E5 use 6/8/11 features respectively. E6 keeps E5's trained classifier and alpha and selects theta on validation. E7 fixes alpha=1.

| model | accuracy | macro_f1 | change_precision | change_recall | change_f1 | exact_destination | stable_accuracy | high_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | 0.90748 | 0.88951 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 1.00000 | 0.00000 |
| E1 | 0.83694 | 0.80233 | 0.28339 | 0.46388 | 0.35184 | 0.41065 | 0.88040 | 0.30493 |
| E2 | 0.88461 | 0.86615 | 0.39601 | 0.26426 | 0.31699 | 0.15589 | 0.95891 | 0.03436 |
| E4 | 0.83518 | 0.80742 | 0.32096 | 0.55894 | 0.40777 | 0.40114 | 0.87943 | 0.30263 |


## 1. Why is Markov accuracy high?

5,159 of 5,685 observed-current test outcomes remain in the same state. The diagonal is the maximum in each saved transition row, so hard predictions are persistence.

## 2. Why does Markov miss changes?

It uses only the discrete current state and always selects that state. Physical proximity and trends do not enter its row selection.

## 3. How many changes does standalone SVM detect?

The Phase 6-compatible calibrated linear SVM detects 244 of 526. Historical raw hard SVM decisions are separate in cross_phase_comparison.csv.

## 4. Does Phase 6 improve the trade-off?

F3 restores stability and accuracy relative to the standalone calibrated SVM but lowers change recall. A trade-off is not dominance.

## 5. Do threshold features help?

E3 test change F1 is 0.4058, versus F3 0.3170; stability declines.

## 6. Do target trends help?

E4 test change F1 is 0.4078, a small increase over E3; validation selects E4.

## 7. Do cross-station features help?

E5 test change F1 is 0.4276, but its validation objective is below E4. General superiority is not established.

## 8. Does the full model beat individual groups?

On the test trade-off E5 is stronger than E3/E4, but not by the locked validation criterion; no test-based reselection was made.

## 9. Are there more false alarms?

Yes: 622 for E4 versus 212 for F3. Whether that is acceptable requires an application-specific false-alarm cost.

## 10. Does exact destination improve?

Yes, from 0.1559 to 0.4011 on actual changes.

## 11. Does stable accuracy decline?

Yes, from 0.9589 to 0.8794.

## 12. Does overall accuracy improve?

No: 0.8846 to 0.8352.

## 13. Does macro-F1 improve?

No: 0.8661 to 0.8074.

## 14. Does calibration improve?

No. Brier, log loss and confidence ECE worsen. Balanced class weighting is a relevant mechanism; probabilities are not claimed calibrated.

## 15. Is there statistical support?

Paired day-block intervals exclude zero for increased recall/change-F1 and for decreased accuracy/stability. This supports both sides of the trade-off under the bootstrap assumptions.

## 16. Is computation justified?

The added logistic models are small and fast; computational cost is modest. Operational justification remains contingent on alarm costs and external validation.

## 17. What should the paper highlight?

Highlight the validation-selected E4 as the proposed method, F3 and Markov as competing trade-offs, and E5 as a prespecified ablation with promising test evidence. Do not claim a universally best model.

## 18. What remains limited?

Single target, historical reuse, calibration deterioration, fixed Markov destination, one holdout, short bootstrap blocks and incomplete original professor requirements.
