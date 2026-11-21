# Phase 5 Targeted Audit Remediation — Issue-to-File Plan

Written **before** any code was changed. Statuses are updated at the end of the file
(`## Final status`) once work has actually been run.

## Package triage (which archive is authoritative)

| Archive | Content | Verdict |
|---|---|---|
| `output.zip` | Full run tree: source, tests, results, logs, predictions, models, figures, **checkpoints (70 pkl)**, `config/` (empty) | **Authoritative** (superset) |
| `Phase5_SVM_Kernel_RFF_Results.zip` | Same tree *without* checkpoints / config, plus top-level `README.md`, `phase5_run.py`, `tests/test_phase5.py` | Distribution wrapper; identical to `output.zip` except `s08_package_and_audit.py` (stale hard-coded numbers) and trailing-newline diffs |
| `Phase_5_Input_FIles.zip` | Immutable Phase 2–4 inputs | Read-only inputs; never modified |

Pristine copies and SHA-256 of every original file are kept in `backup_pre_remediation/`.

## Findings → remediation

| # | Finding (verified against archive) | Files affected | Action |
|---|---|---|---|
| 1 | Stage B: no result CSV, no log in either archive. `apply_frozen_scaler` reads `mean_`/`scale_`; saved scalers contain `mean`/`scale` (verified: keys = mean, scale, var, n_samples_seen). Fallback branch would **fit a scaler on whatever array it is given**. "OvR vs OvO" ablation only toggles `decision_function_shape`, which cannot change `SVC.predict` (always OvO). First-N subsets cover one season only. Absolute Mac default paths. | `source/s04_stage_b_rbf_svc.py` (rewrite), `source/common.py` (new) | Rewrite: shared strict scaler loader (accepts `mean`/`scale`, legacy `mean_`/`scale_`, **raises** if absent, never fits); deterministic temporal-block subsets with coverage/class-shift report; per-run status JSON + registry; child-process fits with enforced wall-time limit; genuine `OneVsRestClassifier(SVC)` vs native OvO ablation plus a demonstration that the old toggle is a no-op; run it. |
| 2 | `audit_summary.json` is produced from **hard-coded literals** (`s08`), which disagree with prediction CSVs (e.g. Linear PCA-C test macro-F1 0.777 vs **0.7907** from CSV; PCA-B 0.7656 vs 0.7614; RFF-HoSVD absent). Report prose carries the same stale values and a wrong `rff_hosvd_B` seed/val-F1 (seed 43 / 0.6980 vs lock: seed 44 / 0.7421). | `source/s08…`, new `source/s10_tables_and_bootstrap.py`, `source/s11_report_and_audit.py` | All aggregate tables recomputed from prediction CSVs by an independent implementation; reconciled against `final_evaluation_scopeAB.csv`; report + audit summary **generated** from those tables (no literals). Old files kept under `superseded_pre_audit/`. |
| 3 | Bootstrap artifact `results/bootstrap_ci.json`: transition-F1 diff CI = **[-0.0170, 0.0316]** (includes 0), macro-F1 diff CI = [0.0054, 0.0513]. Report/audit claim [0.0068, 0.0664] and [0.0264, 0.0601], "statistically robust". Blocks were 24 *row-index* blocks, not timestamp-based. | `source/s07…` (Study 5), `source/s10…` | Quote the artifact correctly; re-run bootstrap with real-timestamp calendar-day blocks (primary) and 7-day blocks (sensitivity), per-block sufficient statistics, several paired comparisons; interpretive language limited to what each interval supports. |
| 4 | RFF final selection = `idxmax` over single-seed validation rows (pca_B → seed 43; hosvd_B → seed 44): best-seed selection, and the pair did not share frequencies. | `source/s06…` (kept unchanged as the **original lock**), new `source/s09_post_audit_revision.py`, `source/common.py` | New pre-declared validation-only rule (D by cross-seed mean; seed = median-ranked common seed of the pair). Original lock preserved verbatim. Revised models flagged as **post-audit revision** (test labels of the original models had already been seen). All 5 seeds retained and reported; descriptive test metrics for all seeds are shown but not used for selection. |
| 5 | Tests contain `/Users/…` absolute paths; no regression tests for scaler keys, leakage-safe fit/transform, report consistency. | `tests/test_phase5_full.py` (rewrite), `tests/conftest.py`, `tests/test_remediation.py` (new) | `--input-dir/--output-dir` options and `PHASE5_*` env vars; explicit skips with reasons when inputs are absent; new regression tests; execute and report actual counts. |
| 6 | `phase5_run.py --stage all` omits Stage B; no skip/resume logic; no revision/bootstrap/report stages. | `phase5_run.py` | `all` runs every stage in order, resumes (never overwrites completed stage outputs unless `--force`), records status in `logs/run_registry.jsonl`. |
| 7 | Calibration: works (train-tail 80/20, multinomial LR on 3 scores), but documentation should state: tail-split rather than multi-fold; Phase 4 bases fitted on full train; calibrator fitted on scores of an 80%-train model but applied to scores of the 100%-train model; class order [0,1,2]; Brier unscaled; EPS 1e-12 clip+renormalise. | `reports/*`, `s09` | Same protocol reused unchanged for revised RFF models; limitations written explicitly. Stage B probabilities left **missing** with reason. |
| 8 | Provenance: no record of what changed. | `manifests/*`, `reports/CHANGELOG_REMEDIATION.md` | Hash-diff original vs final tree; regenerated inventory; locked/revised selection manifests; complete ZIP rebuilt. |

## Will NOT be done
Phases 1–4 untouched; no Phase 6; no Markov/SVM fusion; no train+validation refit; no test-driven selection.
