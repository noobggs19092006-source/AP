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

@pytest.mark.parametrize('value,expected',[(34.999,0),(35,1),(74.999,1),(75,2),(np.nan,-1)])
def test_boundaries(value,expected):assert states([value])[0]==expected


def test_prior_immutability():
    repo=Path(json.loads((ROOT/'config/experiment_spec.json').read_text())['repo'])
    hashes=json.loads((ROOT/'manifests/frozen_prior_phase_hashes.json').read_text())
    assert all(sha(repo/k)==v for k,v in hashes.items())


@pytest.mark.parametrize('split,n,missing',[('validation',4239,42),('test',5720,35)])
def test_alignment_scope_fallback(split,n,missing):
    base=pd.read_csv(ROOT/f'features/{split}_precursors.csv')
    for exp in range(8):
        f=pd.read_csv(ROOT/f'predictions/{split}/E{exp}.csv');assert len(f)==n
        assert f.target_timestamp.equals(base.target_timestamp) and f.y.equals(base.y)
        v=f.current_state<0;assert v.sum()==missing and np.array_equal(v,f.fallback_used)
        assert np.allclose(f.loc[v,['p0','p1','p2']],f.loc[v,['fallback_p0','fallback_p1','fallback_p2']])
        assert np.all(f.loc[v,'pred']==f.loc[v,['p0','p1','p2']].to_numpy().argmax(1))


