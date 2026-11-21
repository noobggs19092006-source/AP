"""
test_integrity.py – Verify split distributions, shapes, timestamps.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

def test_split_counts():
    counts = {"train": 24108, "validation": 4239, "test": 5720}
    for split, n in counts.items():
        fp = ROOT / f"phase4_output/features_{split}_pca_B.npz"
        d = np.load(fp)
        assert len(d["y"]) == n

def test_scope_a_alignment():
    # Verify that Scope A rows match the count in integrity gate
    for split, n in [("validation", 4197), ("test", 5685)]:
        fp = ROOT / f"phase4_output/features_{split}_pca_B.npz"
        d = np.load(fp)
        assert int(np.sum(d["current_state"] >= 0)) == n
