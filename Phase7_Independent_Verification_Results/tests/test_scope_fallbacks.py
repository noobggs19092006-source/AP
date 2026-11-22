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


def test_missing_state_not_transition():
    f=frame([1,0],[1,0],[-1,0]);assert calculate(f,'A')['N']==1 and calculate(f,'B')['N']==2


