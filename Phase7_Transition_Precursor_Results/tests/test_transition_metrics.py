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

def test_wrong_destination():
    m=metrics(np.array([2,0,1]),np.array([1,0,1]),np.array([0,0,0]))
    assert m['change_recall']==1 and m['exact_destination']==.5
    assert np.isclose(m['accuracy'],(2*m['exact_destination']+m['stable_accuracy'])/3)


