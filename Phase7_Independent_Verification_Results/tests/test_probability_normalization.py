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


def test_probability_loss():
    f=frame([0,1,2],[0,1,2],[0,0,1]);f[['p0','p1','p2']]=1/3;m=calculate(f,'A');assert np.isclose(m['brier'],2/3) and np.isclose(m['log_loss'],np.log(3))


