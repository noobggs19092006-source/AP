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

def test_station_specific_inverse_and_lags():
    X=np.zeros((2,24,12,11));X[:,:,:,0]=np.arange(24)[None,:,None]+np.arange(12)[None,None,:]
    mu=np.ones((12,11))*10;sd=np.ones((12,11))*2
    h,m=extract(X,mu,sd,np.array([1,2]))
    assert m[0,23,0]==56 and np.allclose(h[:,6:8],2)
    assert np.allclose(h[:,8:10],2) and np.all(h[:,10]==1)
    X[:,:,0,0]=1000
    h,_=extract(X,mu,sd,np.array([2,2]));assert np.allclose(h[:,8:10],2)


