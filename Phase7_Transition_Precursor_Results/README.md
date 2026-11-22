# Phase 7: Transition-Precursor-Aware Markov–SVM Fusion

## Overview

This implementation extends Phase 6 Markov–SVM fusion with precursor features that provide early warning signals before PM2.5 pollution state transitions.

## Directory Structure

```
Phase7_Transition_Precursor_Results/
├── source/           # Core implementation modules
├── features/         # Precursor features extracted from data
├── models/           # Trained classifier models
├── results/          # Experiment results and metrics
├── predictions/      # Model predictions on test set
├── figures/          # Scientific visualizations
├── diagrams/         # Mathematical diagrams
├── reports/          # Report documents
├── literature/       # Related papers and references
├── manifests/        # Data and model manifests
├── tests/            # Unit tests
└── logs/             # Execution logs
```

## Key Concepts

### PM2.5 Pollution State Classification
- **State 0 (Low)**: PM2.5 < 35 μg/m³
- **State 1 (Moderate)**: 35 ≤ PM2.5 < 75 μg/m³  
- **State 2 (High)**: PM2.5 ≥ 75 μg/m³

### J Metric
```
J = 0.5 × Macro-F1 + 0.5 × Change-F1
```
Balances overall classification accuracy with change detection performance.

### Precursor Features
1. **Threshold Proximity**: Distance to transition boundaries
2. **Target-Station Trends**: Short-term PM2.5 changes
3. **Cross-Station Trends**: Patterns from other monitoring stations

### Fusion Formula
```
q_new = α × q_P + (1 - α) × q_F3
```
Where:
- q_P = precursor-aware logistic probability
- q_F3 = Phase 6 fusion probability (blended change + destination)
- α = precursor weighting parameter

## Experiments

| Experiment | Description |
|------------|-------------|
| E0 | Markov baseline only |
| E1 | Linear SVM only |
| E2 | Phase 5 RBF SVM |
| E3 | Phase 6 F1 Fixed fusion |
| E4 | Phase 6 F2 Adaptive fusion |
| E5 | Phase 6 Transition fusion |
| E6 | Precursor features only |
| E7 | Full precursor-aware fusion |

## Usage

```python
python run_phase7_precursor.py
```

## Results

See `results/phase7_comparison.csv` for full experimental comparison.

## References

See `literature/` directory for related papers.