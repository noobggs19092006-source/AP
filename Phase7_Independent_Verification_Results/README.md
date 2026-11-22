# Independent Phase 7 verification

Run `python -B phase7_audit_run.py` from any directory. The default implementation folder is the sibling `Phase7_Transition_Precursor_Results`. Use `--implementation PATH` to override it. The implementation config identifies the original AP repository; no original files are modified.

The audit independently reconstructs all 24 historical PM2.5 positions from raw station CSVs, including split-local forward fill and training-median replacement. It independently calculates precursor features, original SVM/F3 outputs, logistic equations, fused probabilities, hard decisions, class/event metrics, calibration losses and paired day-block confidence intervals. It does not import the implementation metric functions. Graph data is traced back to raw predictions or explicitly identified historical evidence.

Run `python -B -m pytest tests -q -p no:cacheprovider` after the audit. Results and genuine logs are in results/ and logs/. PASS is used for completed checks only. Missing original professor requirements, an absent original-feature SVM baseline and the separate nested-calibration predictions remain NOT VERIFIED. Historical model fitting is not repeated merely to regenerate plots. Warnings are retained rather than hidden to obtain a clean score.

Both archives are verified by package_results.py in the implementation directory, which performs ZIP CRC and byte-for-byte member checks against the current output files. Source datasets and original large model artifacts are intentionally external dependencies, not duplicated in the archives.
