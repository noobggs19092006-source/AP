import sys,json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from audit_source.metrics_independent import calculate

def frame(y,pred,s):
    f=pd.DataFrame({'y':y,'pred':pred,'current_state':s})
    f[['p0','p1','p2']]=np.eye(3)[pred];return f


def test_validation_and_fallback():
    for name in ['validation_selection_audit','scopeA_scopeB_audit','input_alignment_audit']:
        f=pd.read_csv(ROOT/f'results/{name}.csv');assert (f.status=='PASS').all()


