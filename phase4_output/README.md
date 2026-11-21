# Phase 4 -- PCA/SVD vs shared-basis HoSVD (unsupervised reconstruction)

Status: executed. Completed configurations: A, B, C, B_alt1, B_alt2; not run: none. PCA solver: exact-gram.
Read `PHASE4_RESULTS_AND_MATH.txt` first (math, measured results, caveats, test results).

## Reproduce
```
pip install -r phase4_output/requirements.txt
python phase4_output/phase4_pca_hosvd.py --phase2-zip Air_Pollution_Phase2_Processed_Data.zip --output phase4_output --seed 42 --configs A,B,C,B_alt1,B_alt2
PHASE2_DIR=<extracted phase2_output> python -m pytest -rs -v phase4_output/tests | tee phase4_output/pytest_results.txt
python phase4_output/phase4_pca_hosvd.py --output phase4_output --finalize
```
Other options: `--phase2-dir`, `--configs`, `--batch-size`, `--pca-solver auto|exact-gram|truncated-fallback`, `--max-memory-gb`, `--fallback-max-k`, `--timing-repeats`, `--no-fallback-crosscheck`.
`--pca-solver exact-gram` never falls back silently; `truncated-fallback` is labeled approximate and restricted to k<=`--fallback-max-k`.

## Key files
- `pca_hosvd_comparison.csv`, `compression_storage.csv`, `hosvd_truncation_bound.csv`, `pca_components_spectrum.csv`, `hosvd_mode_spectra.csv`, `phase4_feature_manifest.csv`, `stability_diagnostics.csv`, `phase4_run_status.csv`
- `features_{train,validation,test}_{pca,hosvd}_{A,B,C}.npz`: keys exactly F (N,k) float32, y int8, target_timestamp datetime64[ns], current_state int8, current_pm25 float32 (NaNs kept). `current_pm25` is metadata, NOT part of F.
- `fitted_center_and_factors.npz` (M, full and per-config HoSVD factors, spectra), `fitted_pca_basis_{48,192,576}.npz` (float64 V_k)
- `phase4_metadata.json`, `execution.log`, `figures/`

## Conventions
Features are in Phase-2 standardized units. Flatten order is C (time, station, feature). HoSVD feature index = p*(r_s*r_f)+q*r_f+r. Model parameters are stored and counted as float64; codes are float32.
Scope: unsupervised representation only (no SVM/RFF/Markov/classification). Phase 5 must decide any predictive use; lower reconstruction error does not imply better classification.
