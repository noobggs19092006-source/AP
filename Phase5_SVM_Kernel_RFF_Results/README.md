# Phase 5 — SVM / exact RBF / Random-Fourier-Feature classification (post-audit remediated package)

Spatiotemporal next-hour PM2.5 **state** prediction at Aotizhongxin from Phase 4 PCA / HoSVD features.
States are **research-defined** (Low < 35, Moderate 35–75, High ≥ 75 µg/m³), not official AQI categories.
Phases 1–4 are immutable inputs: this package only reads them and never refits or rewrites them.

> **Status.** This is a remediation of an earlier Phase 5 package after an independent audit.
> Everything in `reports/` and `audit_summary.json` is generated from result files by `source/s11_report_and_audit.py`.
> Nothing here has been independently verified yet. Read `reports/CHANGELOG_REMEDIATION.md` first.

## Layout (the ZIP root is the package root)
```
README.md  phase5_run.py  requirements.txt  audit_summary.json
source/      s01 gate · s02 QP · s03 Stage A · s04 Stage B · s05 Stage C · s06 lock+evaluation · s07 comparative
             s08 packaging/provenance · s09 post-audit revision · s10 tables+bootstrap · s11 report/audit · common.py
config/      experiment_spec.json  grids_and_seeds.json  selection_contract.json
tests/       conftest.py  test_phase5_full.py (14 spec tests)  test_remediation.py  test_qp.py
results/     candidate tables, final metrics (from prediction CSVs), bootstrap, ablations, test_summary.json ...
predictions/ aligned CSVs: original lock (root), revised/, stage_b/   (validation + test per model)
models/      scalers, final models, calibrators, RFF mappings  (pickles: see security note)
figures/  reports/  manifests/  logs/
superseded_pre_audit/   original versions of every file that was replaced (kept for audit; not used)
```
Not included (large / immutable): the Phase 2–4 inputs and the resumable `checkpoints/` directory.
`manifests/input_provenance.json` lists the expected input names and SHA-256 values.

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Tested with Python 3.12, numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, pandas 3.0.6 (remediation environment).
The original Stage A/C/`s06` runs used the author's own environment (Python 3.12–3.14, scikit-learn ≥ 1.5).

## Inputs
Pass the folder holding the Phase 2–4 files (`features_*.npz`, `phase4_feature_manifest.csv`, `markov_predictions_*.csv`, `*_windows.npz`, …):
```bash
python phase5_run.py --input-dir /path/to/Phase5_inputs --stage all
```
No absolute paths exist anywhere in the source or tests. `--output-dir` defaults to the package folder.

## One command / staged runs
`--stage all` runs: `gate → qp → stage_a → stage_b → stage_c → eval → comparative → revision → tables → report → test → finalize → package`.
Single stages: `--stage stage_b`, `--stage revision`, … Useful flags: `--threads`, `--max-ram-gb` (Stage B RAM guard, estimate only),
`--rbf-max-seconds` (enforced per-fit wall-time limit), `--full-exact` (optional full-train exact-RBF pilot), `--force`.

**Checkpointing / resume.** A stage whose primary outputs already exist is skipped (`skipped_existing` in `logs/run_registry.jsonl`);
completed results are never overwritten without `--force`. Stage B resumes per grid cell (`checkpoints/stage_b/*.json`), the revision stage per RFF seed.

## Tests
```bash
python -m pytest tests -v -rs --input-dir /path/to/Phase5_inputs --output-dir .
```
Tests that need the large inputs are **skipped** (with a reason) when `--input-dir` / `PHASE5_INPUT_DIR` is absent.
The last run's actual counts are in `results/test_summary.json` and `logs/pytest_log.txt`.

## What was executed where
| Component | Executed in | Evidence |
|---|---|---|
| Integrity gate, QP (synthetic), Stage A (40 fits), Stage C (30 fits), `s06` lock + evaluation | original run (author's machine) | original logs/results, preserved |
| Stage B grid (36), OvO/OvR ablation, full-train exact pilot | remediation run (1 vCPU sandbox) | `logs/s04_stage_b.log`, `results/stage_b_*.csv` |
| RFF seed rule, revised RFF finals (+calibration), seed sweep, Stage B addendum | remediation run | `logs/s09_post_audit_revision.log` |
| Aggregate tables, reconciliation, bootstrap, report, audit summary | remediation run | `results/final_metrics_from_predictions.csv`, `results/bootstrap_ci.json` |

Original Stage A/C/`s06` outputs were **not re-run**; they are validated by reloading the saved models and reconciling saved predictions with recomputed metrics.

## Security note
`models/*.pkl` are Python pickles; unpickling can execute arbitrary code. Load them only from a source you trust and in a matching scikit-learn version.
Non-executable equivalents: `models/scaler_*.npz` and `models/rff_mapping_*.npz` (plain arrays).

## Scope limits (see report §16)
Phase 4 bases were fitted over the whole training period; calibration is a single 80/20 chronological split; Stage B has no probabilities;
Stage A/B/C optima sit on grid edges in places; RFF/exact dimension controls (d = 48, 576) and alternative pairs were not run;
no Phase 6 work and no Markov/SVM fusion is included.
