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

def test_scaler_and_deterministic_refit():
    from sklearn.base import clone
    f=pd.read_csv(ROOT/'features/train_precursors.csv');lock=json.loads((ROOT/'config/selected_configuration.json').read_text());w=lock['winners']['E4']
    obj=pickle.load(open(ROOT/f'models/{w["classifier"]}.pkl','rb'));cols=obj['features'];v=f.current_state>=0;X=f.loc[v,cols].to_numpy();y=(f.loc[v,'y']!=f.loc[v,'current_state']).astype(int)
    assert np.allclose(obj['pipeline'][0].mean_,X.mean(0))
    again=clone(obj['pipeline']).fit(X,y)
    assert np.allclose(again.predict_proba(X[:50]),obj['pipeline'].predict_proba(X[:50]),atol=1e-12)
    assert again.predict_proba(X[:50]).shape==(50,2)


def test_bootstrap_pairing_and_reproducibility():
    from source.statistical_analysis import bootstrap
    f=pd.read_csv(ROOT/'predictions/test/E4.csv').iloc[:120].copy()
    a=bootstrap({'E0':f,'E2':f,'E4':f},'E4',reps=12,seed=42);b=bootstrap({'E0':f,'E2':f,'E4':f},'E4',reps=12,seed=42)
    pd.testing.assert_frame_equal(a,b)
    assert np.all(a.loc[a.reference!='absolute',['lower','upper']].to_numpy()==0)


