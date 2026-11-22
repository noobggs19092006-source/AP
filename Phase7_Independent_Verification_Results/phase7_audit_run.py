"""Independent calculations from prediction CSVs, frozen inputs and stored models."""
import argparse,hashlib,json,pickle,sys,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from audit_source.metrics_independent import calculate
from audit_source.file_integrity import sha
from audit_source.precursor_math_audit import independent_features
from audit_source.fusion_math_audit import logistic_from_parameters
from audit_source.bootstrap_audit import percentile_contrast
HERE=Path(__file__).resolve().parent
def main(root):
    for d in ['results','reports','tests','logs','manifests']:(HERE/d).mkdir(exist_ok=True)
    checks=[];reconcile=[];features=[];prob=[];alignment=[];fallback=[]
    def check(name,ok,detail='',status=None):checks.append(dict(check=name,status=status or ('PASS' if ok else 'FAIL'),detail=detail))
    config=json.loads((root/'config/selected_configuration.json').read_text());repo=Path(json.loads((root/'config/experiment_spec.json').read_text())['repo'])
    frozen=json.loads((root/'manifests/frozen_prior_phase_hashes.json').read_text());integrity=[]
    for name,digest in frozen.items():integrity.append(dict(path=name,expected=digest,actual=sha(repo/name),status='PASS' if sha(repo/name)==digest else 'FAIL'))
    check('Prior research immutability',all(r['status']=='PASS' for r in integrity),f'{len(integrity)} files')
    pd.DataFrame(integrity).to_csv(HERE/'results/artifact_integrity.csv',index=False)
    (HERE/'manifests/audited_input_hashes.json').write_text(json.dumps(frozen,indent=2))
    from audit_source.alignment_audit import audit_raw
    from audit_source.baseline_audit import verify as verify_baseline
    raw_audit=audit_raw(root,repo);raw_audit.to_csv(HERE/'results/physical_pm25_reconstruction.csv',index=False)
    check('All histories including independent imputation reconstruction',(raw_audit.status=='PASS').all())
    baseline=verify_baseline(root,repo);baseline.to_csv(HERE/'results/phase6_baseline_reconciliation.csv',index=False)
    check('Independent original model and F3 reproduction',(baseline.status=='PASS').all())
    pars=np.load(repo/'Phase 2/phase2_output/training_preprocessing_parameters.npz')
    for split in ['train','validation','test']:
        z=np.load(repo/f'Phase 2/phase2_output/{split}_windows.npz');f=pd.read_csv(root/f'features/{split}_precursors.csv')
        X=z['X'];physical=X[:,:,:,0].astype(float)*pars['std'][:,0]+pars['mean'][:,0]
        independent=independent_features(physical,z['current_state'])
        names=['d35','d75','dmin','state_low','state_moderate','state_high','v3','v6','cross_mean','cross_max','fraction_rising']
        err=float(abs(independent-f[names].to_numpy()).max());features.append(dict(split=split,max_error=err));check(f'{split}: independent feature formulas',err<1e-9)
        good=np.array_equal(pd.to_datetime(f.target_timestamp).to_numpy(),z['target_timestamp']) and np.array_equal(f.y,z['y']) and np.array_equal(f.current_state,z['current_state'])
        check(f'{split}: timestamps and targets',good);alignment.append(dict(split=split,rows=len(f),status='PASS' if good else 'FAIL'))
        del X,physical
    recorded=pd.read_csv(root/'results/final_evaluation.csv')
    for split in ['validation','test']:
        feat=pd.read_csv(root/f'features/{split}_precursors.csv')
        for exp in [f'E{i}' for i in range(8)]:
            f=pd.read_csv(root/f'predictions/{split}/{exp}.csv');v=f.current_state>=0;s=f.current_state.to_numpy();ids=np.flatnonzero(v)
            p=f[['p0','p1','p2']].to_numpy();dest=f[['d0','d1','d2']].to_numpy()
            check(f'{split}/{exp}: simplex',np.isfinite(p).all() and p.min()>=0 and np.allclose(p.sum(1),1,atol=1e-12))
            ok=np.allclose(p[~v],f.loc[~v,['fallback_p0','fallback_p1','fallback_p2']]);check(f'{split}/{exp}: fallback',ok)
            fallback.append(dict(split=split,model=exp,missing_current=int((~v).sum()),status='PASS' if ok else 'FAIL'))
            if exp in config['winners']:
                w=config['winners'][exp];obj=pickle.load(open(root/f'models/{w["classifier"]}.pkl','rb'));pipeline=obj['pipeline']
                q=logistic_from_parameters(feat[obj['features']].to_numpy(),pipeline)
                err=float(abs(q-f.q_precursor).max());check(f'{split}/{exp}: explicit logistic equation',err<1e-10)
                qnew=w['alpha']*q+(1-w['alpha'])*f.q_f3.to_numpy();rebuilt=qnew[ids,None]*dest[ids];rebuilt[np.arange(len(ids)),s[ids]]=1-qnew[ids]
                perr=float(abs(rebuilt-p[ids]).max());check(f'{split}/{exp}: independent fusion equation',perr<1e-10)
                hard=rebuilt.argmax(1) if 'theta' not in w else np.where(qnew[ids]>=w['theta'],dest[ids].argmax(1),s[ids])
                check(f'{split}/{exp}: hard rule',np.array_equal(hard,f.pred.to_numpy()[ids]));prob.append(dict(split=split,model=exp,logistic_max_error=err,fusion_max_error=perr))
            for scope in ['A','B']:
                met=calculate(f,scope);saved=recorded[(recorded.model==exp)&(recorded.split==split)&(recorded.scope==scope)].iloc[0]
                error=max(abs(float(saved[k])-float(value)) for k,value in met.items());reconcile.append(dict(split=split,model=exp,scope=scope,max_error=error,**met));check(f'{split}/{exp}/{scope}: independent metrics',error<1e-10)
                check(f'{split}/{exp}/{scope}: destination <= change recall',met['exact_destination']<=met['change_recall']+1e-12)
                if scope=='A':check(f'{split}/{exp}: weighted accuracy identity',np.isclose(met['accuracy'],((met['TP']+met['FN'])*met['exact_destination']+(met['TN']+met['FP'])*met['stable_accuracy'])/met['N']))
    grid=pd.read_csv(root/'results/precursor_fusion_grid.csv');selection=[]
    for exp,group in [('E3','threshold'),('E4','temporal'),('E5','all'),('E7','all')]:
        g=grid[grid.group==group]
        if exp=='E7':g=g[g.alpha==1]
        win=g.sort_values(['J','macro_f1','n_features','C','weighted','id'],ascending=[False,False,True,True,True,True]).iloc[0]
        ok=win.id==config['winners'][exp]['id'];check(f'{exp}: validation-only selection',ok);selection.append(dict(model=exp,winner=win.id,status='PASS' if ok else 'FAIL'))
    tg=pd.read_csv(root/'results/transition_threshold_grid.csv');win=tg.sort_values(['J','macro_f1','n_features','C','weighted','id'],ascending=[False,False,True,True,True,True]).iloc[0]
    check('E6: validation-only threshold',win.id==config['winners']['E6']['id'])
    # Independent paired resampling; same calendar block IDs and RNG contract.
    selected=config['selected_experiment'];names=list(dict.fromkeys(['E0','E2',selected]));frames={k:pd.read_csv(root/f'predictions/test/{k}.csv') for k in names}
    dates=pd.to_datetime(frames[selected].target_timestamp).dt.floor('D');blocks=[np.flatnonzero(dates==d) for d in dates.unique()]
    keys=['accuracy','macro_f1','change_precision','change_recall','change_f1','exact_destination','stable_accuracy','high_f1']
    rng=np.random.default_rng(42);samples={k:[] for k in names};boot=pd.read_csv(root/'results/bootstrap_confidence_intervals.csv');reps=int(boot.resamples.iloc[0])
    # Scalar event metrics only for speed; probability quality is audited above.
    for _ in range(reps):
        ix=np.concatenate([blocks[i] for i in rng.integers(0,len(blocks),len(blocks))])
        for name,f in frames.items():
            b=f.iloc[ix];b=b[b.current_state>=0];y=b.y.to_numpy();pred=b.pred.to_numpy();s=b.current_state.to_numpy();div=lambda a,b:a/b if b else 0.
            a=y!=s;c=pred!=s;t=np.sum(a&c);fp=np.sum(~a&c);fn=np.sum(a&~c)
            mf=np.mean([div(2*np.sum((y==k)&(pred==k)),np.sum(y==k)+np.sum(pred==k)) for k in range(3)])
            ha=(s!=2)&(y==2);hb=(s!=2)&(pred==2)
            samples[name].append([np.mean(y==pred),mf,div(t,t+fp),div(t,t+fn),div(2*t,2*t+fp+fn),div(np.sum(a&(y==pred)),np.sum(a)),div(np.sum(~a&(y==pred)),np.sum(~a)),div(2*np.sum(ha&hb),np.sum(ha)+np.sum(hb))])
    br=[]
    for _,r in boot.iterrows():
        bounds=percentile_contrast(samples,r.model,r.reference,keys.index(r.metric));err=max(abs(bounds-[r.lower,r.upper]));br.append(dict(model=r.model,reference=r.reference,metric=r.metric,max_error=err))
    check('Independent paired bootstrap',max(r['max_error'] for r in br)<1e-10,f'{reps} calendar-day resamples')
    for name,rows in [('input_alignment_audit',alignment),('precursor_feature_reconciliation',features),('model_probability_reconciliation',prob),('validation_selection_audit',selection),('final_metrics_reconciliation',reconcile),('scopeA_scopeB_audit',fallback),('bootstrap_reconciliation',br)]:pd.DataFrame(rows).to_csv(HERE/f'results/{name}.csv',index=False)
    # Figure data is verified in the publication audit; absent coverage is not a pass.
    index=root/'figures/figure_index.csv'
    if index.exists():
        figures=pd.read_csv(index);fr=[]
        for _,r in figures.iterrows():
            ok=(root/r.figure).is_file() and (root/r.data).is_file()
            fr.append(dict(category=r.category,figure=r.figure,status='PASS' if ok else 'FAIL',detail='File/data presence; numerical source audit in publication checks'))
        from audit_source.figure_audit import verify as verify_figures
        numerical=verify_figures(root,repo);numerical.to_csv(HERE/'results/graph_verification_matrix.csv',index=False)
        check('20 graph categories: numerical source consistency',len(numerical)==20 and (numerical.status=='PASS').all())
        artists=pd.read_csv(root/'figures/artist_value_checks.csv');check('Plotted bar artist values',(artists.max_error<1e-12).all())
        check('Six rendered diagrams',len(list((root/'diagrams').glob('*.png')))==6)
    else:check('20 graph categories',False,'Publication generator not run',status='NOT VERIFIED')
    for name in ['Phase7_Transition_Precursor_Results','Phase7_Independent_Verification_Results']:
        z=root.parent/f'{name}.zip'
        if z.exists():
            with zipfile.ZipFile(z) as archive:check(f'{name}: ZIP CRC',archive.testzip() is None)
        else:check(f'{name}: ZIP CRC',False,'Archive not yet packaged',status='NOT VERIFIED')
    check('Original professor-issued P2-P7 rubric',False,'Working descriptions only; authoritative original not supplied',status='NOT VERIFIED')
    reproduction=pd.read_csv(root/'results/reproducibility_results.csv')
    check('Second full execution: identical models and predictions',(reproduction.status=='PASS').all(),f'{len(reproduction)} artifact hashes')
    check('Nested calibration isolation claim',False,'Preserved nested_calibration.py fits calibrators on in-sample SVM training scores; separate experiment not reused',status='WARNING')
    check('Original 3168-feature SVM baseline',False,'Not found in supplied research; no substitute invented',status='NOT VERIFIED')
    check('Nested-calibration row-level comparison',False,'Historical experiment lacks saved predictions',status='NOT VERIFIED')
    check('Probability calibration',False,'Selected balanced model worsens probability losses; no post-test changes',status='WARNING')
    check('Literature primary access',False,'One foundational bootstrap primary source blocked; matrix records access depth',status='WARNING')
    check('Legacy precursor source preserved',False,'Unused pre-existing source/precursor_features.py is retained; correct entry point imports physical_precursors',status='WARNING')
    pd.DataFrame(checks).to_csv(HERE/'results/final_verification_matrix.csv',index=False)
    counts=pd.Series([r['status'] for r in checks]).value_counts().to_dict();(HERE/'logs/audit_execution.log').write_text(json.dumps(counts,indent=2));print(counts,flush=True)
    (HERE/'reports/INDEPENDENT_PHASE7_AUDIT.md').write_text('# Independent Phase 7 numerical audit\n\n'+json.dumps(counts,indent=2)+'\n\nMetrics were recalculated from predictions without importing implementation metrics. Feature formulas, logistic equations, probability reconstruction, selection and paired bootstrap were independently evaluated. See results/final_verification_matrix.csv for individual evidence.\n',encoding='utf8')
    if counts.get('FAIL',0):sys.exit(1)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--implementation',type=Path,default=HERE.parent/'Phase7_Transition_Precursor_Results');main(p.parse_args().implementation)
