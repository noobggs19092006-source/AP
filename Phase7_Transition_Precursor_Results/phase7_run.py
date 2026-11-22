"""Reproducible precursor experiment. Existing repository and legacy files are read-only.
Run from any directory: python phase7_run.py --repo ../AP
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse, hashlib, json, pickle, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from source.physical_precursors import extract, states, NAMES, GROUPS
from source.precursor_fusion import f3,predict
from source.evaluation import metrics
from source.statistical_analysis import bootstrap
from source.integrity_checks import sha,dump
from source.timestamp_alignment import align
from source.ablation_experiments import rank
from source.transition_classifier import new_classifier
from source.load_prior_artifacts import load_linear

OUT=Path(__file__).resolve().parent

def main(repo,reps):
    started=time.perf_counter(); repo=repo.resolve()
    for d in ['config','features','models','results','predictions/validation','predictions/test','figures','diagrams','reports','literature','manifests','tests','logs']:(OUT/d).mkdir(parents=True,exist_ok=True)
    log=open(OUT/'logs/execution.log','w',encoding='utf8',buffering=1)
    def say(s):print(s,flush=True);log.write(s+'\n')
    # Freeze all existing research bytes, not just files used by the model.
    freeze=OUT/'manifests/frozen_prior_phase_hashes.json'
    if freeze.exists():
        hashes=json.loads(freeze.read_text())
        assert all((repo/k).is_file() and sha(repo/k)==v for k,v in hashes.items()),'Prior inputs changed since checkpoint'
    else:
        hashes={p.relative_to(repo).as_posix():sha(p) for p in sorted(repo.rglob('*')) if p.is_file() and not any(x in p.parts for x in ['.git','__pycache__','.pytest_cache'])}
        dump(freeze,hashes)
    say(f'Frozen and checked {len(hashes)} prior files.')
    spec={'repo':str(repo),'seed':42,'C':[.1,1.,10.],'class_weight':[None,'balanced'],'alpha':[0,.25,.5,.75,1.],
          'theta':[.2,.3,.4,.5,.6,.7],'selection':'validation Scope A: max J, macro F1, fewer features, smaller C, unweighted, ID',
          'bootstrap':{'calendar_block_hours':24,'reps':reps,'seed':42},'python':sys.version,'numpy':np.__version__,'sklearn':sklearn.__version__,'platform':platform.platform()}
    dump(OUT/'config/experiment_spec.json',spec);dump(OUT/'config/training_grid.json',{k:spec[k] for k in ['C','class_weight']})
    dump(OUT/'config/fusion_grid.json',{k:spec[k] for k in ['alpha','theta']})
    dump(OUT/'config/precursor_features.json',{'order':NAMES,'groups':GROUPS,'scaling':'train Scope A only','lags':[23,20,17],'station':0,'channel':0})
    params=np.load(repo/'Phase 2/phase2_output/training_preprocessing_parameters.npz')
    assert params['station'][0]=='Aotizhongxin' and params['feature'][0]=='PM2.5'
    P=pd.read_csv(repo/'Air_Pollution_Phase3_Results/phase3_output/markov_transition_matrix.csv').filter(regex='^next_').to_numpy()
    assert P.shape==(3,3) and np.allclose(P.sum(1),1)
    # Raw data checks and imputation provenance, without regenerating original artifacts.
    raw=[]
    for station in params['station']:
        paths=list((repo/'beijing+multi+site+air+quality+data').rglob(f'PRSA_Data_{station}_20130301-20170228.csv'))
        assert len(paths)==1
        f=pd.read_csv(paths[0]); ix=pd.to_datetime(f[['year','month','day','hour']])
        assert len(ix)==35064 and not ix.duplicated().any()
        raw.append(pd.Series(f['PM2.5'].to_numpy(),index=ix,name=station))
    raw=pd.concat(raw,axis=1); assert not raw.index.duplicated().any() and raw.index.to_series().diff().dropna().eq(pd.Timedelta(hours=1)).all()
    data={}; integrity=[]; reconstruction=[]
    for split,n in [('train',24108),('validation',4239),('test',5720)]:
        path=repo/f'Phase 2/phase2_output/{split}_windows.npz'
        with open(path,'rb') as f:assert not f.read(100).startswith(b'version https://git-lfs')
        with np.load(path) as z:
            y=z['y'].astype(int);s=z['current_state'].astype(int);ts=z['target_timestamp'];X=z['X']
            assert X.shape==(n,24,12,11) and len(np.unique(ts))==n and np.all(np.diff(ts)>np.timedelta64(0,'h'))
            h,m=extract(X,params['mean'],params['std'],s);del X
        assert np.array_equal(states(raw.loc[pd.to_datetime(ts),'Aotizhongxin']),y)
        assert np.array_equal(states(raw.loc[pd.to_datetime(ts)-pd.Timedelta(hours=1),'Aotizhongxin']),s)
        masks=[]
        for lag,index in [(0,23),(3,20),(6,17)]:
            original=raw.loc[pd.to_datetime(ts)-pd.Timedelta(hours=lag+1)].to_numpy()
            valid=np.isfinite(original);error=float(abs(original[valid]-m[:,index,:][valid]).max())
            assert error<.001
            masks.append(~valid)
            reconstruction.append(dict(split=split,lag_hours=lag,max_absolute_error=error,observed_values=int(valid.sum())))
        frame=pd.DataFrame(h,columns=NAMES);frame.insert(0,'target_timestamp',pd.to_datetime(ts));frame['y']=y;frame['current_state']=s
        frame['target_history_imputed']=np.stack(masks)[:,:,0].any(0)
        frame['cross_history_imputed']=np.stack(masks[:2])[:,:,1:].any(axis=(0,2))
        frame.to_csv(OUT/f'features/{split}_precursors.csv',index=False)
        data[split]={'h':h,'y':y,'s':s,'ts':ts}
        integrity.append(dict(split=split,shape=[n,24,12,11],valid_current=int((s>=0).sum()),missing_current=int((s<0).sum()),changes=int(((s>=0)&(s!=y)).sum())))
        del m
    assert integrity[1]['missing_current']==42 and integrity[2]['missing_current']==35 and integrity[2]['changes']==526
    pd.DataFrame(reconstruction).to_csv(OUT/'results/physical_reconstruction.csv',index=False)
    dump(OUT/'results/input_integrity.json',{'status':'PASS','splits':integrity,'station_order':params['station'].tolist(),'channel_order':params['feature'].tolist()})
    pd.DataFrame([{'feature':name,'unit':'ug/m3' if i<3 else ('ug/m3/hour' if i in [6,7,8,9] else 'dimensionless'),'train_min':data['train']['h'][:,i].min(),'train_max':data['train']['h'][:,i].max()} for i,name in enumerate(NAMES)]).to_csv(OUT/'features/feature_dictionary.csv',index=False)
    say('Verified physical histories, hourly timestamps, missing states and split counts.')
    modeldir=repo/'Phase5_SVM_Kernel_RFF_Results/models'
    model,cal,scale=load_linear(repo)
    assert list(model.classes_)==list(cal.classes_)==[0,1,2]
    reconciliation=[]
    for split in ['validation','test']:
        d=data[split];f=np.load(repo/f'phase4_output/features_{split}_pca_C.npz');assert np.array_equal(f['target_timestamp'],d['ts'])
        z=(f['F']-scale['mean'])/scale['scale'];p=cal.predict_proba(model.decision_function(z))
        saved=align(pd.read_csv(repo/f'Phase5_SVM_Kernel_RFF_Results/predictions/final_linear_best_overall_{split}.csv'),d['ts'])
        error=float(abs(p-saved[['prob_0','prob_1','prob_2']].to_numpy()).max());assert error<1e-6,error
        d['svm']=p;d['markov'],d['qf'],d['dest'],d['f3']=f3(d['s'],p,P)
        reconciliation.append(dict(check='saved_SVM_probabilities',split=split,max_error=error,status='PASS'))
        if split=='test':
            old=align(pd.read_csv(repo/'Phase6_Markov_SVM_Fusion_Results/predictions/test/test_predictions_merged.csv'),d['ts']);valid=d['s']>=0
            err=float(abs(d['f3'][valid]-old[['p_f3_0','p_f3_1','p_f3_2']].to_numpy()[valid]).max());assert err<1e-6,err
            reconciliation.append(dict(check='F3_ScopeA_probabilities',split=split,max_error=err,status='PASS'))
    pd.DataFrame(reconciliation).to_csv(OUT/'results/baseline_reconciliation.csv',index=False)
    say('Original saved SVM and F3 probability reproduction passed.')
    train=data['train'];val=data['validation'];valid=train['s']>=0
    classifiers={};grid=[];cost=[]
    for group,cols in GROUPS.items():
        for C in spec['C']:
            for weight in spec['class_weight']:
                mid=f'{group}_C{C}_{weight}';t=time.perf_counter()
                clf=new_classifier(C,weight)
                clf.fit(train['h'][valid][:,cols],(train['y'][valid]!=train['s'][valid]).astype(int))
                fit=time.perf_counter()-t;classifiers[mid]=(clf,cols)
                pickle.dump({'pipeline':clf,'columns':cols,'features':[NAMES[i] for i in cols]},open(OUT/f'models/{mid}.pkl','wb'))
                qp=clf.predict_proba(val['h'][:,cols])[:,1]
                cost.append(dict(model=mid,fit_seconds=fit,model_bytes=(OUT/f'models/{mid}.pkl').stat().st_size,n_features=len(cols)))
                for alpha in spec['alpha']:
                    t=time.perf_counter();p,pred,q=predict(qp,val['qf'],val['dest'],val['s'],val['svm'],alpha)
                    row=metrics(val['y'],pred,val['s'],p);row.update(id=f'{mid}_a{alpha}',classifier=mid,group=group,C=C,weighted=weight is not None,alpha=alpha,n_features=len(cols),runtime_seconds=time.perf_counter()-t,split='validation',scope='A');grid.append(row)
    winners={}
    for exp,group in [('E3','threshold'),('E4','temporal'),('E5','all')]:winners[exp]=min([r for r in grid if r['group']==group],key=rank)
    winners['E7']=min([r for r in grid if r['group']=='all' and r['alpha']==1],key=rank)
    threshold=[];base=winners['E5'];clf,cols=classifiers[base['classifier']];qp=clf.predict_proba(val['h'][:,cols])[:,1]
    for theta in spec['theta']:
        p,pred,q=predict(qp,val['qf'],val['dest'],val['s'],val['svm'],base['alpha'],theta)
        row={**base,**metrics(val['y'],pred,val['s'],p),'id':base['id']+f'_theta{theta}','theta':theta};threshold.append(row)
    winners['E6']=min(threshold,key=rank)
    selected=min(winners,key=lambda k:rank(winners[k]))
    locked={'selection_split':'validation','selection_scope':'A','test_labels_used_for_selection':False,'selected_experiment':selected,'winners':winners,'config_sha256':sha(OUT/'config/experiment_spec.json'),'input_manifest_sha256':sha(freeze)}
    # This manifest is written before any new model test predictions or test scores.
    dump(OUT/'config/selected_configuration.json',locked);dump(OUT/'manifests/model_selection_manifest.json',locked)
    pd.DataFrame(grid).to_csv(OUT/'results/precursor_fusion_grid.csv',index=False)
    pd.DataFrame([r for r in grid if r['alpha']==1]).to_csv(OUT/'results/precursor_model_grid.csv',index=False)
    pd.DataFrame(threshold).to_csv(OUT/'results/transition_threshold_grid.csv',index=False)
    say(f'Locked validation-selected experiment: {selected}, {winners[selected]["id"]}.')
    results=[];frames={}
    for split in ['validation','test']:
        d=data[split]
        for exp in ['E0','E1','E2','E3','E4','E5','E6','E7']:
            t=time.perf_counter();qp=np.full(len(d['y']),np.nan);q=d['qf'].copy()
            if exp in ['E0','E1','E2']:
                p=d[{'E0':'markov','E1':'svm','E2':'f3'}[exp]].copy();pred=p.argmax(1)
                v=d['s']>=0;q[v]=1-p[np.flatnonzero(v),d['s'][v]]
            else:
                w=winners[exp];clf,cols=classifiers[w['classifier']];qp=clf.predict_proba(d['h'][:,cols])[:,1]
                p,pred,q=predict(qp,d['qf'],d['dest'],d['s'],d['svm'],w['alpha'],w.get('theta'))
            elapsed=time.perf_counter()-t
            f=pd.DataFrame({'target_timestamp':pd.to_datetime(d['ts']),'y':d['y'],'current_state':d['s'],'pred':pred,'probability_argmax':p.argmax(1),'fallback_used':d['s']<0,'q_precursor':qp,'q_f3':d['qf'],'q_new':q})
            for k in range(3):f[f'p{k}']=p[:,k];f[f'd{k}']=d['dest'][:,k];f[f'fallback_p{k}']=d['svm'][:,k]
            f.to_csv(OUT/f'predictions/{split}/{exp}.csv',index=False)
            for scope in ['A','B']:
                row=metrics(d['y'],pred,d['s'],p,scope);row.update(model=exp,split=split,scope=scope,prediction_seconds=elapsed,selected=exp==selected);results.append(row)
            if split=='test':frames[exp]=f
    result=pd.DataFrame(results);result.to_csv(OUT/'results/final_evaluation.csv',index=False)
    result.to_csv(OUT/'results/feature_ablation.csv',index=False)
    for name,cols in {'transition_metrics':['TP','FP','TN','FN','change_precision','change_recall','change_f1'], 'exact_destination_metrics':['exact_destination','stable_accuracy'], 'classwise_metrics':[c for c in result if c.startswith('class')], 'confusion_matrices':[c for c in result if c.startswith('cm_')], 'probability_quality':['brier','log_loss','ece'], 'high_entry_metrics':['high_TP','high_FP','high_FN','high_precision','high_recall','high_f1']}.items():
        result[['model','split','scope','N']+cols].to_csv(OUT/f'results/{name}.csv',index=False)
    pd.DataFrame(cost).to_csv(OUT/'results/computational_cost.csv',index=False)
    test=result.query("split=='test' and scope=='A'").set_index('model');diff=[]
    for base in ['E0','E1','E2']:
        for key in ['accuracy','macro_f1','change_precision','change_recall','change_f1','exact_destination','stable_accuracy','high_f1']:
            diff.append(dict(model=selected,baseline=base,metric=key,delta=test.loc[selected,key]-test.loc[base,key],percentage_points=100*(test.loc[selected,key]-test.loc[base,key])))
    pd.DataFrame(diff).to_csv(OUT/'results/model_improvement_vs_baselines.csv',index=False)
    say('Running 1,000 paired calendar-day bootstrap resamples.')
    bootstrap({k:frames[k] for k in dict.fromkeys(['E0','E2',selected])},selected,reps).to_csv(OUT/'results/bootstrap_confidence_intervals.csv',index=False)
    features=pd.read_csv(OUT/'features/test_precursors.csv');features['actual_change']=(features.y!=features.current_state)&(features.current_state>=0)
    for exp in ['E0','E2',selected]:features[f'missed_{exp}']=features.actual_change&(frames[exp].pred.to_numpy()==features.current_state)
    features.to_csv(OUT/'results/missed_transition_analysis.csv',index=False)
    features.groupby('actual_change')[NAMES].agg(['mean','std','median']).to_csv(OUT/'results/transition_feature_statistics.csv')
    robustness=[]
    for label,mask in [('observed_precursors',~(features.target_history_imputed|features.cross_history_imputed)),('imputed_precursors',features.target_history_imputed|features.cross_history_imputed)]:
        for exp in ['E0','E2',selected]:
            f=frames[exp].loc[mask];row=metrics(f.y.to_numpy(),f.pred.to_numpy(),f.current_state.to_numpy(),f[['p0','p1','p2']].to_numpy());robustness.append(dict(model=exp,stratum=label,**row))
    pd.DataFrame(robustness).to_csv(OUT/'results/robustness_results.csv',index=False)
    dump(OUT/'manifests/input_provenance.json',{'repo':str(repo),'frozen_files':len(hashes),'software':spec,'prior_phase_unchanged':all(sha(repo/k)==v for k,v in hashes.items()),'elapsed_seconds':time.perf_counter()-started})
    say('Core execution complete; publication outputs and independent audit are separate steps.')
    log.close()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,default=OUT.parent/'AP');ap.add_argument('--bootstrap',type=int,default=1000);args=ap.parse_args();main(args.repo,args.bootstrap)
