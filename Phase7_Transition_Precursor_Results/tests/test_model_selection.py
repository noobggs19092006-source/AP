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

def test_validation_selection():
    g=pd.read_csv(ROOT/'results/precursor_fusion_grid.csv');lock=json.loads((ROOT/'config/selected_configuration.json').read_text())
    assert not lock['test_labels_used_for_selection'] and set(g.split)=={'validation'}
    for exp,group in [('E3','threshold'),('E4','temporal'),('E5','all')]:
        assert min(g[g.group==group].to_dict('records'),key=rank)['id']==lock['winners'][exp]['id']


def test_ablations_and_binary_training_rows():
    lock=json.loads((ROOT/'config/selected_configuration.json').read_text());train=pd.read_csv(ROOT/'features/train_precursors.csv');n=sum(train.current_state>=0)
    groups=[]
    for exp,count in [('E3',6),('E4',8),('E5',11)]:
        w=lock['winners'][exp];obj=pickle.load(open(ROOT/f'models/{w["classifier"]}.pkl','rb'));assert len(obj['columns'])==count and obj['pipeline'][0].n_samples_seen_==n;groups.append(obj['columns'])
    assert groups[0]!=groups[1]!=groups[2]


