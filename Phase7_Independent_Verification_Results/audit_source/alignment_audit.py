"""Independently reconstruct original hourly PM2.5 and imputation policy."""
import numpy as np
import pandas as pd

def audit_raw(root,repo):
    params=np.load(repo/'Phase 2/phase2_output/training_preprocessing_parameters.npz');stations=params['station'];raw=[]
    for station in stations:
        path=next((repo/'beijing+multi+site+air+quality+data').rglob(f'PRSA_Data_{station}_20130301-20170228.csv'))
        f=pd.read_csv(path);t=pd.to_datetime(f[['year','month','day','hour']]);raw.append(pd.Series(f['PM2.5'].to_numpy(),index=t,name=station))
    raw=pd.concat(raw,axis=1);rows=[]
    state=lambda a:np.where(np.isfinite(a),np.where(a<35,0,np.where(a<75,1,2)),-1)
    borders=[('train','2013-03-01','2016-01-01'),('validation','2016-01-01','2016-07-01'),('test','2016-07-01','2017-03-01')]
    for split,start,end in borders:
        f=raw[(raw.index>=start)&(raw.index<end)];filled=f.ffill(limit=6).fillna(dict(zip(stations,params['median'][:,0])))
        z=np.load(repo/f'Phase 2/phase2_output/{split}_windows.npz');ts=pd.to_datetime(z['target_timestamp']);X=z['X']
        errors=[]
        for index in range(24):
            time=ts-pd.Timedelta(hours=24-index);expected=filled.loc[time].to_numpy();actual=X[:,index,:,0].astype(float)*params['std'][:,0]+params['mean'][:,0]
            errors.append(float(np.max(np.abs(expected-actual))))
        good=np.array_equal(state(raw.loc[ts,'Aotizhongxin'].to_numpy()),z['y']) and np.array_equal(state(raw.loc[ts-pd.Timedelta(hours=1),'Aotizhongxin'].to_numpy()),z['current_state'])
        rows.append(dict(split=split,max_absolute_error=max(errors),all_24_lags=True,imputation_policy='split-local forward fill <=6 h then saved training median',target_and_current_states_match=good,status='PASS' if max(errors)<.001 and good else 'FAIL'))
        del X
    return pd.DataFrame(rows)
