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

@pytest.mark.parametrize('alpha',[0,.25,.5,.75,1])
def test_probability_math(alpha):
    s=np.array([0,1,2,-1]);sv=np.array([[.6,.3,.1],[.1,.7,.2],[.2,.3,.5],[.3,.6,.1]])
    P=np.array([[.9,.09,.01],[.05,.9,.05],[.01,.09,.9]])
    m,q,d,p=f3(s,sv,P);pn,hard,qn=predict(np.ones(4)*.4,q,d,s,sv,alpha)
    assert np.allclose(d[:3].sum(1),1) and np.all(d[np.arange(3),s[:3]]==0)
    assert np.allclose(pn.sum(1),1) and pn.min()>=0 and pn.max()<=1
    assert np.array_equal(pn[-1],sv[-1])
    if alpha==0:assert np.allclose(pn,p)
    if alpha==1:assert np.allclose(qn[:3],.4)


def test_threshold_preserves_probabilities():
    s=np.array([0,1]);sv=np.array([[.8,.1,.1],[.2,.7,.1]]);P=np.array([[.9,.05,.05],[.1,.8,.1],[.1,.1,.8]])
    _,q,d,_=f3(s,sv,P);qp=np.array([.6,.6]);a,ha,_=predict(qp,q,d,s,sv,1);b,hb,_=predict(qp,q,d,s,sv,1,.5)
    assert np.array_equal(a,b) and np.any(ha!=hb)


