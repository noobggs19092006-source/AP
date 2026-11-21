"""
markov_probabilities.py – Build p_M from Phase-3 train-fitted transition matrix P.
For observed current state i: p_M,t = row i of P, class order [0,1,2].
Scope B missing-current-state fallback: log warning, return None per-row; caller must provide explicit fallback (see phase6_run.py).
"""
import numpy as np
import pandas as pd


class MarkovProbabilities:
    def __init__(self, P_path):
        df_P = pd.read_csv(P_path)
        # CSV has label header: current_state,next_0_low,next_1_moderate,next_2_high
        self.P = df_P[[c for c in df_P.columns if c.startswith("next_")]].values.astype(np.float64)
        assert self.P.shape == (3, 3), f"P shape {self.P.shape} != (3,3)"
        if not np.allclose(self.P.sum(axis=1), 1.0, atol=1e-10):
            raise ValueError("P rows do not sum to 1")

    def get_probs(self, current_state: np.ndarray):
        """Return (N,3) probabilities for all rows where cs >= 0. Rows with cs==-1 are left as NaN."""
        N = len(current_state)
        out = np.full((N, 3), np.nan, dtype=np.float64)
        mask = current_state >= 0
        if not mask.any():
            return out
        idx = current_state[mask].astype(int)
        if idx.min() < 0 or idx.max() > 2:
            raise ValueError(f"Invalid current states in eligible subset: {np.unique(idx)}")
        out[mask] = self.P[idx]
        return out

    def hard_predictions(self, current_state: np.ndarray):
        """Markov argmax predictions for eligible rows only."""
        p = self.get_probs(current_state)
        with np.errstate(invalid="ignore", divide="ignore"):
            hp = np.where(np.isfinite(p).all(axis=1), np.argmax(p, axis=1), -1)
        return hp
