"""Reconcile every category's numerical plot data with its underlying evidence."""
import numpy as np
import pandas as pd
from .metrics_independent import calculate

def verify(root,repo):
    index=pd.read_csv(root/'figures/figure_index.csv');out=[]
    rawmetrics=pd.read_csv(root.parent/'Phase7_Independent_Verification_Results/results/final_metrics_reconciliation.csv')
    test=rawmetrics.query("split=='test' and scope=='A'").set_index('model')
    for r in index.itertuples():
        d=pd.read_csv(root/r.data);ok=True;detail=''
        if r.category in [1,2,3,4,5,10,13,19]:
            for row in d.itertuples():
                for col in ['accuracy','macro_f1','change_precision','change_recall','change_f1','exact_destination','stable_accuracy','high_f1']:
                    if col in d:ok &= np.isclose(getattr(row,col),test.loc[row.model,col])
            if r.category==13:
                for row in d.itertuples():
                    f=pd.read_csv(root/f'predictions/test/{row.model}.csv');f=f[f.current_state>=0]
                    for i in range(3):
                        for j in range(3):ok &= getattr(row,f'cm_{i}{j}')==sum((f.y==i)&(f.pred==j))
            detail='Independent metrics from raw prediction CSVs'
        elif r.category==6:
            b=pd.read_csv(root/'results/bootstrap_confidence_intervals.csv').query("reference=='E2'").set_index('metric')
            for row in d.itertuples():
                ok &= np.isclose(row.delta,test.loc[row.model,row.metric]-test.loc['E2',row.metric]) and np.isclose(row.lower,b.loc[row.metric,'lower']) and np.isclose(row.upper,b.loc[row.metric,'upper'])
            detail='Independent metric differences and independently reconciled bootstrap intervals'
        elif r.category in [7,8,9]:
            f=pd.read_csv(root/'results/missed_transition_analysis.csv');f=f[f.current_state>=0]
            if r.category==7:
                f['bin']=pd.cut(f.dmin,[0,5,10,20,40,80,np.inf],include_lowest=True)
                selected=d.columns.tolist();g=f.groupby('bin',observed=True)
                ok=np.array_equal(g.actual_change.sum().to_numpy(),d.actual_changes) and np.array_equal(g.missed_E2.sum().to_numpy(),d.missed_F3)
                # The selected model is identified in the lock, not guessed from graph title.
                import json
                name=json.loads((root/'config/selected_configuration.json').read_text())['selected_experiment']
                ok &= np.array_equal(g[f'missed_{name}'].sum().to_numpy(),d.missed_selected)
            else:
                cols=['v3','v6'] if r.category==8 else ['cross_mean','cross_max','fraction_rising']
                observations=pd.read_csv(root/f'figures/data/{r.category:02d}_observations.csv')
                ok=np.allclose(observations[cols],f[cols]) and np.array_equal(observations.actual_change,f.actual_change)
                for _,row in d.iterrows():
                    g=f[f.actual_change==row.actual_change]
                    for col in cols:
                        for agg in ['mean','median','std']:ok &= np.isclose(row[f'{col}_{agg}'],getattr(g[col],agg)())
            detail='Observation-level feature grouping and summary recalculation'
        elif r.category in [11,12]:
            grid=pd.read_csv(root/('results/precursor_fusion_grid.csv' if r.category==11 else 'results/transition_threshold_grid.csv')).set_index('id')
            for row in d.itertuples():
                for c in ['J','macro_f1','change_recall','change_precision','stable_accuracy']:ok &= np.isclose(getattr(row,c),grid.loc[row.id,c])
            detail='Locked validation candidate table; model selection separately audited'
        elif r.category==14:
            for row in d.itertuples():
                f=pd.read_csv(root/f'predictions/test/{row.model}.csv');f=f[f.current_state>=0];p=f[['p0','p1','p2']].to_numpy()
                confidence=p.max(1) if row.kind=='class' else f.q_new.to_numpy();truth=(p.argmax(1)==f.y) if row.kind=='class' else (f.y!=f.current_state)
                mask=np.minimum((confidence*10).astype(int),9)==row.bin
                ok &= int(mask.sum())==row.N and np.isclose(confidence[mask].mean(),row.predicted) and np.isclose(np.asarray(truth)[mask].mean(),row.observed)
            detail='Reliability bins independently rebuilt'
        elif r.category==15:
            import json
            lock=json.loads((root/'config/selected_configuration.json').read_text());cost=pd.read_csv(root/'results/computational_cost.csv').set_index('model')
            for row in d.itertuples():
                name=lock['winners'][row.model]['classifier'];ok &= row.model_bytes==(root/f'models/{name}.pkl').stat().st_size and np.isclose(row.fit_seconds,cost.loc[name,'fit_seconds']) and np.isclose(row.J,.5*(test.loc[row.model,'macro_f1']+test.loc[row.model,'change_f1']))
            detail='Artifact sizes and recorded new-model timing; single-run timing not independently benchmarked'
        elif r.category==16:
            original=pd.read_csv(repo/'phase4_output/pca_hosvd_comparison.csv').query("split=='test'")
            ok=np.allclose(d.relative_reconstruction_error,original.relative_reconstruction_error) and np.allclose(d.relative_reconstruction_error**2,1-d.retained_energy,atol=1e-7)
            detail='Historical source equality and energy/error identity; original decompositions not retrained'
        elif r.category==17:
            original=pd.read_csv(repo/'Phase5_SVM_Kernel_RFF_Results/results/stage_a_best_per_config.csv').set_index('config')
            for row in d.itertuples():ok &= np.isclose(row.val_scopeA_macro_f1,original.loc[row.config,'val_scopeA_macro_f1'])
            detail='Historical validation table equality; original grid not retrained'
        elif r.category in [18,20]:
            for _,row in d.iterrows():
                if row.source=='new prediction CSV':
                    base=test.loc[row.model]
                elif 'Phase5' in row.source:
                    f=pd.read_csv(repo/row.source);f=f.rename(columns={'hard_pred':'pred','prob_0':'p0','prob_1':'p1','prob_2':'p2'})
                    # Independent hard metrics do not require calibrated probabilities.
                    if not np.isfinite(f[['p0','p1','p2']]).all().all():
                        f[['p0','p1','p2']]=np.eye(3)[f.pred]
                    base=calculate(f,'A')
                else:
                    f=pd.read_csv(repo/row.source);name=row.model.replace('Phase6_','');f=f.rename(columns={'true_next_state':'y','observed_current_state':'current_state',f'pred_{name}':'pred',**{f'p_{name}_{k}':f'p{k}' for k in range(3)}});base=calculate(f,'A')
                for c in ['accuracy','macro_f1','change_f1']:ok &= np.isclose(row[c],base[c])
            detail='Cross-phase hard metrics independently recomputed from source predictions'
        out.append(dict(category=r.category,figure=r.figure,status='PASS' if ok else 'FAIL',detail=detail))
    return pd.DataFrame(out)
