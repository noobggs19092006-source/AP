"""Leakage-aware Phase 2 preprocessing for the Beijing PRSA 12-station dataset.
Run: python phase2_build.py --zip PRSA_Data_20130301-20170228.zip --out phase2_output
Requires numpy, pandas. Generates separate compressed train/val/test tensors.
Label convention: 0=low (<35), 1=moderate (35 to <75), 2=high (>=75) ug/m3, for next-hour instantaneous PM2.5 (NOT an official AQI).
"""
import argparse, json, zipfile, pathlib, warnings
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

STATIONS = ['Aotizhongxin','Changping','Dingling','Dongsi','Guanyuan','Gucheng','Huairou','Nongzhanguan','Shunyi','Tiantan','Wanliu','Wanshouxigong']
FEATURES = ['PM2.5','PM10','SO2','NO2','CO','O3','TEMP','PRES','DEWP','RAIN','WSPM']
BORDERS = [('train','2013-03-01','2016-01-01'), ('validation','2016-01-01','2016-07-01'), ('test','2016-07-01','2017-03-01')]
WINDOW=24

def states(a):
    y=np.full(np.shape(a),-1,dtype=np.int8)
    finite=np.isfinite(a)
    y[finite & (a<35)]=0
    y[finite & (a>=35) & (a<75)]=1
    y[finite & (a>=75)]=2
    return y

def main(zip_path,out):
    out=pathlib.Path(out);out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        paths={s:next((n for n in z.namelist() if n.endswith(f'PRSA_Data_{s}_20130301-20170228.csv')),None) for s in STATIONS}
        assert all(paths.values()), f'Missing station CSV: {[s for s in STATIONS if not paths[s]]}'
        frames=[]
        for s in STATIONS:
            with z.open(paths[s]) as f: df=pd.read_csv(f)
            assert df['station'].eq(s).all(),s
            df['timestamp']=pd.to_datetime(df[['year','month','day','hour']])
            assert not df.timestamp.duplicated().any(),s
            frames.append(df.set_index('timestamp').sort_index())
    times=frames[0].index
    assert all(d.index.equals(times) for d in frames)
    raw=np.stack([d[FEATURES].to_numpy(dtype='float64') for d in frames],axis=1)  # T,S,F
    target_raw=raw[:,0,0].copy()
    report={'timestamps':len(times),'stations':STATIONS,'features':FEATURES,'shape_raw':list(raw.shape), 'thresholds':{'low':'PM2.5 < 35','moderate':'35 <= PM2.5 < 75','high':'PM2.5 >= 75','units':'ug/m3','caveat':'Research-defined instantaneous hourly PM2.5 bins; not official AQI/24-hour guideline levels.'},'window_hours':WINDOW,'target':'Aotizhongxin PM2.5 state in hour immediately after the window','notes':['Sequential forward-fill limited to 6 hours within each split (never backward-filled).','Any still-missing measurements replaced by median for that station-feature fitted on training split only.','Standardization fitted on imputed training hours only, separately by station-feature.','Target labels derived from original unfilled PM2.5 and unobserved targets excluded.','Windows formed separately within chronological splits: no cross-boundary history.','Current state derived from raw PM2.5 at last historical hour; -1 if missing, so exclude from Markov fusion evaluation.','Wind direction excluded from the 11 numeric features; can be added later as controlled ablation.','All reported compressed sizes reflect on-disk NPZ files; uncompressed logical array sizes are reported separately.'],'splits':{}}
    train_mask=(times>=pd.Timestamp(BORDERS[0][1]))&(times<pd.Timestamp(BORDERS[0][2]))
    train=raw[train_mask]
    train_filled=pd.DataFrame(train.reshape(-1,len(STATIONS)*len(FEATURES))).ffill(limit=6).to_numpy().reshape(train.shape)
    med=np.nanmedian(train_filled,axis=0)
    assert np.isfinite(med).all()
    train_final=np.where(np.isfinite(train_filled),train_filled,med[None,:,:])
    mean=train_final.mean(axis=0); std=train_final.std(axis=0)
    std_safe=np.where(std>1e-8,std,1.0)
    np.savez_compressed(out/'training_preprocessing_parameters.npz',median=med,mean=mean,std=std_safe,station=np.array(STATIONS),feature=np.array(FEATURES))
    meta=[]
    for split,start,end in BORDERS:
        mask=(times>=pd.Timestamp(start))&(times<pd.Timestamp(end))
        ts=times[mask]
        subset=raw[mask]
        # forward fill on the time axis, independently for all station-feature series
        arr=pd.DataFrame(subset.reshape(len(ts),-1)).ffill(limit=6).to_numpy().reshape(subset.shape)
        after_ffill=int(np.isnan(arr).sum())
        arr=np.where(np.isfinite(arr),arr,med[None,:,:])
        standardized=((arr-mean)/std_safe).astype(np.float32)
        # sliding_window_view: (N,S,F,W) -> (N,W,S,F)
        X=np.moveaxis(sliding_window_view(standardized,WINDOW,axis=0)[:-1],-1,1)
        y_all=states(target_raw[mask][WINDOW:])
        current_all=states(target_raw[mask][WINDOW-1:-1])
        valid=(y_all>=0)
        X=np.ascontiguousarray(X[valid])
        y=y_all[valid]
        current=current_all[valid]
        y_times=ts[WINDOW:][valid]
        current_pm=target_raw[mask][WINDOW-1:-1][valid]
        assert X.shape[1:]==(24,12,11)
        assert X.shape[0]==y.shape[0]==len(y_times)
        assert np.isfinite(X).all()
        path=out/f'{split}_windows.npz'
        np.savez_compressed(path,X=X,y=y,target_timestamp=y_times.to_numpy(dtype='datetime64[ns]'),current_state=current,current_pm25=current_pm.astype(np.float32))
        counts=np.bincount(y,minlength=3)
        missing_targets=int((~valid).sum())
        # timestamps and states in lightweight human readable csv
        meta.extend({'split':split,'target_timestamp':str(t),'label':int(v),'current_state':int(c)} for t,v,c in zip(y_times,y,current))
        report['splits'][split]={'start':str(ts[0]),'end':str(ts[-1]),'raw_hours':len(ts),'possible_windows':len(ts)-WINDOW,'excluded_missing_target':missing_targets,'usable_samples':int(len(y)),'class_counts':{'low':int(counts[0]),'moderate':int(counts[1]),'high':int(counts[2])},'class_percentages':{'low':round(100*counts[0]/len(y),2),'moderate':round(100*counts[1]/len(y),2),'high':round(100*counts[2]/len(y),2)},'missing_current_state':int((current<0).sum()),'nan_inputs_after_forward_fill_before_median':after_ffill,'tensor_shape':list(X.shape),'raw_float32_MB':round(X.nbytes/1e6,2),'compressed_npz_MB':round(path.stat().st_size/1e6,2),'file':path.name}
        print(split,report['splits'][split],flush=True)
        del X, standardized
    pd.DataFrame(meta).to_csv(out/'window_labels_and_timestamps.csv',index=False)
    (out/'phase2_summary.json').write_text(json.dumps(report,indent=2))
    print('Saved to',out,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--zip',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    main(args.zip,args.out)
