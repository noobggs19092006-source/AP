"""Paired calendar-day block bootstrap, including gaps from missing target labels."""
import numpy as np
import pandas as pd
from source.evaluation import metrics

KEYS=['accuracy','macro_f1','change_precision','change_recall','change_f1','exact_destination','stable_accuracy','high_f1']

def bootstrap(frames,selected,reps=1000,seed=42):
    base=frames[selected]; groups=pd.to_datetime(base.target_timestamp).dt.floor('D')
    blocks=[np.flatnonzero(groups==day) for day in groups.unique()]
    rng=np.random.default_rng(seed); values={k:[] for k in frames}
    for _ in range(reps):
        ix=np.concatenate([blocks[j] for j in rng.integers(len(blocks),size=len(blocks))])
        for name,f in frames.items():
            met=metrics(f.y.to_numpy()[ix],f.pred.to_numpy()[ix],f.current_state.to_numpy()[ix])
            values[name].append([met[k] for k in KEYS])
    rows=[]
    for name,a in values.items():
        a=np.asarray(a)
        for j,key in enumerate(KEYS):
            lo,hi=np.quantile(a[:,j],[.025,.975]); rows.append(dict(model=name,reference='absolute',metric=key,lower=lo,upper=hi,resamples=reps,seed=seed))
            if name==selected:
                for ref in ['E0','E2']:
                    lo,hi=np.quantile(a[:,j]-np.asarray(values[ref])[:,j],[.025,.975])
                    rows.append(dict(model=name,reference=ref,metric=key,lower=lo,upper=hi,resamples=reps,seed=seed))
    return pd.DataFrame(rows)
