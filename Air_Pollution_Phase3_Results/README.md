# Phase 3 -- First-order Markov model and sequential baselines (Aotizhongxin PM2.5 state)

Files: `phase3_markov.py` (CLI), `markov_core.py` (pure math/metrics), `tests/test_markov_core.py`,
`requirements.txt`, `pytest.ini`, `phase3_output/` (all results). The large Phase 2 NPZ arrays are NOT in the ZIP.
Only `y`, `target_timestamp`, `current_state`, `current_pm25` are read; `X` is never loaded. No cloud services, no randomness.

## Setup and run

macOS / Linux:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python phase3_markov.py --phase2-dir phase2_output --out phase3_output --seed 42
python -m pytest -q
```
Windows (PowerShell):
```powershell
py -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python phase3_markov.py --phase2-dir phase2_output --out phase3_output --seed 42
python -m pytest -q
```
Flags: `--timing-repeats N` (default 25), `--no-zip`, `--zip-only` (rebuild `Air_Pollution_Phase3_Results.zip`).
Run from the folder containing `phase2_output/` (unzip `Air_Pollution_Phase2_Processed_Data.zip` there).
The script stops with exit code 2 and a clear message if any Phase 2 count, key, shape, dtype or split bound differs from the contract.

## Conventions (fixed before evaluation)
* P: unsmoothed MLE from the 23,976 TRAIN pairs with observed `current_state`; fitted once, made read-only, SHA-256 checked after fit / evaluation / diagnostics.
* Train prior / majority: from all 24,108 TRAIN `y` labels. Majority ties -> smallest index.
* Markov tie rule: if the current state is in the argmax set predict it, else the minimum index; ties decided by exact integer count equality.
* Scope A `observed_current_only` (identical rows for all models); Scope B `full_split_with_prior_fallback` (missing state -> TRAIN majority / prior).
* Brier = unnormalised sum over 3 classes (0..2). Log loss: clip eps=1e-12, renormalise; not reported for persistence (one-hot, uncalibrated).
* Balanced accuracy = mean recall over classes with support>0; the number of contributing classes is a CSV column.
* Adequacy diagnostics use timestamp-exact triples (raw PRSA CSV was not supplied, so Option 2 is not implemented). Block bootstrap (optional) is not implemented.
