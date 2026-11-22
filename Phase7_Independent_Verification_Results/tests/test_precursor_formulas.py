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


@pytest.mark.parametrize('filename,column,tolerance',[
    ('final_metrics_reconciliation.csv','max_error',1e-10),
    ('precursor_feature_reconciliation.csv','max_error',1e-9),
    ('model_probability_reconciliation.csv','fusion_max_error',1e-10),
    ('model_probability_reconciliation.csv','logistic_max_error',1e-10),
    ('bootstrap_reconciliation.csv','max_error',1e-10),
    ('physical_pm25_reconstruction.csv','max_absolute_error',.001),
    ('phase6_baseline_reconciliation.csv','f3_max_error',1e-8)])
def test_reconciliation(filename,column,tolerance):
    f=pd.read_csv(ROOT/'results'/filename);assert len(f)>0 and (f[column]<tolerance).all()


