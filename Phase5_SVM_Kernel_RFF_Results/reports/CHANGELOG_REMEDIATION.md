# Phase 5 remediation changelog

Companion to `REMEDIATION_PLAN.md` (written before any change). Exact per-file status (unchanged / modified / added / removed)
is machine-generated in `manifests/provenance_changes.json` from SHA-256 hashes taken **before** any modification
(`manifests/original_output_sha256.txt`). Original versions of every replaced file are in `superseded_pre_audit/`.

## Authoritative package
`output.zip` is a strict superset of `Phase5_SVM_Kernel_RFF_Results.zip` (adds `checkpoints/` and an empty `config/`). Apart from trailing-newline
differences the only content difference was `source/s08_package_and_audit.py`, which in the wrapper zip held **older** hard-coded numbers than in
`output.zip`; both archives shipped the same stale `audit_summary.json`. Work was done on the `output.zip` tree and the wrapper-level
`phase5_run.py`/`README.md` were rewritten.

## Audit findings → what was done → evidence

| # | Finding | Verified? | Remediation | Evidence |
|---|---|---|---|---|
| 1 | Stage B had no outputs/logs; scaler key mismatch; fallback fit; absolute paths | Yes: saved scalers contain `mean`,`scale`,`var`,`n_samples_seen`; old code read `mean_`/`scale_`; fallback would fit on the passed array | `s04` rewritten; strict `common.load_frozen_scaler`; per-cell resumable runs with enforced wall-time limit; temporal-block subsets; genuine OvR ablation | `results/stage_b_*.csv`, `logs/s04_stage_b.log`, `manifests/STAGE_B_SELECTION_validation_only.json` |
| 1b | (new) "OvR vs OvO" ablation was a no-op | Yes: predictions identical under `decision_function_shape` toggle | Replaced with `OneVsRestClassifier(SVC)` vs native OvO; no-op demonstrated | `results/stage_b_ovr_ovo_ablation.csv` |
| 1c | (new) "first N rows" subsets covered only the earliest training period | By construction | Systematic temporal blocks spanning all 34 training months; class shift reported | `results/stage_b_subsets.csv` |
| 2 | Report / audit summary inconsistent with predictions | Yes: Linear PCA-C test macro-F1 is 0.7907 from the CSV vs 0.777 in the old summary; other values also differ; `s08` contained literals | Independent recomputation from prediction CSVs, reconciliation (max diff 3.9e-16 vs original `final_evaluation_scopeAB.csv`); report and audit summary generated from files; conditional prose | `results/final_metrics_from_predictions.csv`, `results/reconciliation_vs_final_evaluation.csv`, `source/s11_report_and_audit.py` |
| 2b | (new) Old report's kernel-approximation figures (e.g. 0.201/0.103/0.073 relative error) did not match `rff_kernel_approx.csv` | Yes | Table regenerated from the CSV | report §4 |
| 3 | Bootstrap interval claims | Yes: artifact transition-F1 CI = [−0.0170, +0.0316] (includes 0), macro-F1 CI = [+0.0054, +0.0513]; old report/summary claimed 0.0068–0.0664 and 0.0264–0.0601 and "statistically robust" | Claims withdrawn; bootstrap re-run with real-timestamp calendar blocks (24 h, 168 h), several pairs; prose conditional on whether intervals include zero | `results/bootstrap_ci.json`, `results/bootstrap_ci_table.csv`, report §12 |
| 4 | RFF final selection used best single validation seed | Yes: both original picks were the best single validation seed for their representation; the two used different seeds | Original lock untouched; all 5 seeds kept; new validation-only rule (D by cross-seed mean; median-ranked common seed); **labelled post-audit, not a pre-test lock**; descriptive 5-seed test sweep | `manifests/REVISED_SELECTION_post_audit.json`, `results/rff_seed_*.csv`, `predictions/revised/` |
| 5 | Non-portable tests | Yes (`/Users/...`) | `conftest.py` with `--input-dir/--output-dir`/env vars; explicit skips; rewritten 14 spec tests exercising real code; 20+ regression tests | `tests/`, `results/test_summary.json`, `logs/pytest_log*.txt` |
| 6 | `--stage all` omitted Stage B | Yes | New stage order incl. Stage B, real-subset QP, revision, tables, report, tests, packaging; skip-if-complete; run registry | `phase5_run.py`, `logs/run_registry.jsonl` |
| 7 | Calibration documentation | — | Same train-only 80/20 protocol reused for revised models; limitations documented (single split; Phase 4 bases fitted on whole training period; class order; Brier/clip conventions) | report §10 |
| 8 | Provenance/packaging | — | Hash-diff vs original tree, inventory, complete ZIP in spec layout | `manifests/provenance_changes.json`, `manifests/artifacts_inventory.json` |

## Additional findings discovered during remediation (not in the audit list)
* **Selection rule:** the documented 0.005 tie rule was never implemented in the original selection (plain arg-max). Applying it would change the chosen C for several Stage A representations (`results/stage_a_tie_rule_audit.csv`). The original lock was **not** altered.
* **Spec gap closed:** the 60–150-sample training-only real-data QP check was missing; added as `s02b` (`results/qp_real_subset.json`).
* **Original provenance manifest** stores the author's absolute local paths in its `path` fields (recorded data only; no code reads them).
* **First test run had 2 failures, both defects in newly written tests** (assumed a `skip_reason` column exists when nothing is skipped; flagged the report's own quotation of the withdrawn claim). Tests were corrected, not deleted; the first-run log is kept as `logs/pytest_log_run1_2_failures.txt` and `results/test_summary_run1.json`.
* **Process note:** the first launch of the revision stage was killed by the sandbox after the PCA seeds (no traceback; most likely memory/time limits). It was restructured to bound memory and resumed from its per-seed checkpoints; both partial logs are in `logs/`.

## What was regenerated vs preserved
* **Preserved byte-for-byte (verified by hash):** all Stage A / Stage C / `s06` result CSVs, original prediction CSVs, original models and scalers, original lock manifest, math chapter, QP synthetic results.
* **New:** Stage B, revision, tables/bootstrap, report generation, real-subset QP, tests, config, README.
* **Regenerated:** report, audit summary, bootstrap artifact, figures that carry final-model results (`markov_vs_svm_transitions`, `high_entry_comparison`), inventories.
* **Not re-executed:** original Stage A (40 fits), Stage C (30 fits), `s06` — validated by reload-and-compare only (reloading `final_linear_pca_B.pkl` reproduces saved test predictions exactly; HoSVD-B RFF refit in this environment differs from the original in a couple of test predictions, see report §11).

## Remaining limitations / unexecuted
See report §16 and `audit_summary.json → planned_not_executed`. Nothing here is independently verified.
