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


def test_detection_not_destination():
    f=frame([2,0,1],[1,0,1],[0,0,0]);m=calculate(f,'A')
    assert m['change_recall']==1 and m['exact_destination']==.5 and m['accuracy']==2/3


