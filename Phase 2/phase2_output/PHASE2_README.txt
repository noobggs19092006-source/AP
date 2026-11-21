PHASE 2 — SPATIOTEMPORAL AIR-POLLUTION PREDICTION
Reproducible preprocessing, chronological splits, and tensor construction

DATA AND TASK
12 Beijing PRSA stations, common 35,064-hour timeline (2013-03-01 to 2017-02-28).
Primary target: Aotizhongxin instantaneous PM2.5 state one hour AFTER a historical 24-hour observation window.
Shape per sample: (24 hours, 12 stations, 11 numeric measurement channels) = 3168 float32 entries.
Station order: Aotizhongxin, Changping, Dingling, Dongsi, Guanyuan, Gucheng, Huairou, Nongzhanguan, Shunyi, Tiantan, Wanliu, Wanshouxigong
Feature order: PM2.5, PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, WSPM
Wind direction is excluded from the primary numeric tensor.

STATE DEFINITIONS (RESEARCH CHOICE; NOT AN OFFICIAL AQI)
Low = PM2.5 <35 ug/m3; Moderate = 35<=PM2.5<75 ug/m3; High = PM2.5>=75 ug/m3.
These are instantaneous hourly categories. They must NOT be described as WHO limits or official AQI grades.
Threshold sensitivity (e.g. alternate cuts or training-quantile-based bins) should be explored as an ablation.

SPLITTING AND LEAKAGE PREVENTION
Train: 2013-03-01 through 2015-12-31; validation: 2016-01-01 through 2016-06-30; test: 2016-07-01 through 2017-02-28.
Train-only station-by-feature medians, means and standard deviations are stored in training_preprocessing_parameters.npz.
At each split: forward-fill within that split, at most six consecutive missing hours, never backfill. Remaining missing inputs use the train median; station-feature standardization uses train means/stds.
Targets are ALWAYS generated from original observed PM2.5 before any imputation; unknown target hours are removed.
No history window crosses a split boundary. First 24 hours of each split have no target samples under this convention.
The current-state side-channel is -1 if last-hour PM2.5 is unobserved, so it must be excluded (or separately modelled) for Markov fusion.
Potential caveat: input PM2.5 may itself be imputed; carry missingness indicators in a later ablation if necessary.

ACTUAL COUNTS:
TRAIN: 24,108 valid windows / 24,840 candidate windows; excluded targets=732
  low=7,872 (32.65%), moderate=5,832 (24.19%), high=10,404 (43.16%)
  current-state missing: 132; window tensor: [24108, 24, 12, 11]; logical float32 X size: 305.5 MB; compressed NPZ: 10.45 MB
VALIDATION: 4,239 valid windows / 4,344 candidate windows; excluded targets=105
  low=1,848 (43.6%), moderate=1,114 (26.28%), high=1,277 (30.13%)
  current-state missing: 42; window tensor: [4239, 24, 12, 11]; logical float32 X size: 53.72 MB; compressed NPZ: 1.87 MB
TEST: 5,720 valid windows / 5,808 candidate windows; excluded targets=88
  low=2,142 (37.45%), moderate=1,202 (21.01%), high=2,376 (41.54%)
  current-state missing: 35; window tensor: [5720, 24, 12, 11]; logical float32 X size: 72.48 MB; compressed NPZ: 2.44 MB

CLASS DISTRIBUTION DRIFT
The training prevalence of the high state is 43.16%, versus 30.13% in validation and 41.54% in testing.
Use balanced accuracy, macro-F1, per-class precision/recall, confusion matrices and class-specific transition accuracy.
Do not select thresholds or hyperparameters on the test labels.

DATA FILES
train_windows.npz, validation_windows.npz, test_windows.npz: arrays X (float32), y (int8), target_timestamp, current_state, current_pm25.
training_preprocessing_parameters.npz: station-feature train median, mean, std, station order, feature order.
window_labels_and_timestamps.csv: human-readable target time, label and observed current-state label.
phase2_summary.json: machine-readable dataset and counts.
phase2_build.py: rerun from the original user-provided ZIP.

VALIDATION AND FUTURE STEPS
For each split, check X.shape[1:]==(24,12,11), np.isfinite(X).all(), and y categories in {0,1,2}.
Next Phase 3: chronological persistence/Markov baselines, then train-fitted multilinear SVD and matched-dimensional PCA.
Train the compression bases ONLY on training data; transform validation and test with fixed bases.
Compare raw linear/RBF SVM, PCA+SVM, HoSVD+SVM, HoSVD+RFF+linear SVM.
HoSVD reconstruction bound on training tensor is not automatically a test-set bound.

RUN
python phase2_build.py --zip PRSA_Data_20130301-20170228.zip --out phase2_output
