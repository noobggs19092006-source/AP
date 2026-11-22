import sys,json,pickle
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from source.physical_precursors import extract,states
from source.precursor_fusion import f3,predict
from source.evaluation import metrics
from phase7_run import rank,sha

def test_baseline_reproduction():
    f=pd.read_csv(ROOT/'results/baseline_reconciliation.csv');assert (f.max_error<1e-6).all()


def test_graph_artist_and_full_rerun():
    a=pd.read_csv(ROOT/'figures/artist_value_checks.csv');assert (a.max_error<1e-12).all()
    b=pd.read_csv(ROOT/'results/reproducibility_results.csv');assert len(b)==34 and (b.status=='PASS').all()


