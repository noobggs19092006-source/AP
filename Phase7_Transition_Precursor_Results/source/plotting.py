"""Publication plots with exported numerical source tables and artist-value checks."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from source.evaluation import metrics
from source.cross_phase_comparison import annotate

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130,'savefig.dpi':200})

def generate(root,repo):
    figs=root/'figures';(figs/'data').mkdir(exist_ok=True)
    index=[];artist_checks=[]
    allmet=pd.read_csv(root/'results/final_evaluation.csv');test=allmet.query("split=='test' and scope=='A'").copy()
    selected=json.loads((root/'config/selected_configuration.json').read_text())['selected_experiment']
    frames={f'E{i}':pd.read_csv(root/f'predictions/test/E{i}.csv') for i in range(8)}
    cross=test.copy();cross['source']='new prediction CSV';cross['verification']='recomputed';cross['phase']='Phase 7 controlled experiment'
    for path in sorted((repo/'Phase5_SVM_Kernel_RFF_Results/predictions').rglob('*test.csv')):
        f=pd.read_csv(path);f.target_timestamp=pd.to_datetime(f.target_timestamp)
        f=f.set_index('target_timestamp').loc[pd.to_datetime(frames['E0'].target_timestamp)].reset_index()
        assert np.array_equal(f.y,frames['E0'].y) and np.array_equal(f.current_state,frames['E0'].current_state)
        pred=f.hard_pred.to_numpy();p=f[['prob_0','prob_1','prob_2']].to_numpy()
        if not np.isfinite(p).all():p=None
        row=metrics(f.y.to_numpy(),pred,f.current_state.to_numpy(),p)
        row.update(model=path.stem.replace('_test',''),split='test',scope='A',phase='Phase 5',source=path.relative_to(repo).as_posix(),verification='metrics recomputed from saved hard predictions')
        cross=pd.concat([cross,pd.DataFrame([row])],ignore_index=True)
    old=pd.read_csv(repo/'Phase6_Markov_SVM_Fusion_Results/predictions/test/test_predictions_merged.csv');old.target_timestamp=pd.to_datetime(old.target_timestamp)
    old=old.set_index('target_timestamp').loc[pd.to_datetime(frames['E0'].target_timestamp)].reset_index()
    s=frames['E0'].current_state.to_numpy();y=frames['E0'].y.to_numpy();v=s>=0
    for model in ['f1','f2','f3']:
        p=old[[f'p_{model}_{k}' for k in range(3)]].to_numpy();p[~v]=frames['E1'].loc[~v,['p0','p1','p2']]
        row=metrics(y,p.argmax(1),s,p);row.update(model='Phase6_'+model,phase='Phase 6',split='test',scope='A',source='Phase6_Markov_SVM_Fusion_Results/predictions/test/test_predictions_merged.csv',verification='recomputed Scope A; feature-only fallback outside scope')
        cross=pd.concat([cross,pd.DataFrame([row])],ignore_index=True)
    # Nested experiment has no row-level predictions in the available package.
    nested=pd.read_csv(repo/'Phase7_Nested_CV_Calibration/results/phase7_comparison.csv')
    nested.to_csv(root/'results/nested_calibration_historical_unverified.csv',index=False)
    cross=pd.concat([cross,pd.DataFrame([dict(model='RBF_Nested_Calibration',phase='Separate existing calibration experiment',split='unverified',scope='unverified',verification='NOT VERIFIED: no row-level predictions; historical table saved separately',source='Phase7_Nested_CV_Calibration/results/phase7_comparison.csv')])],ignore_index=True)
    cross=annotate(cross,repo,root)
    cross.to_csv(root/'results/cross_phase_comparison.csv',index=False)
    def save(num,title,data,fig):
        stem=f'{num:02d}_'+''.join(c.lower() if c.isalnum() else '_' for c in title).strip('_')
        data.to_csv(figs/f'data/{stem}.csv',index=False)
        fig.suptitle(title,fontsize=15,fontweight='bold');fig.tight_layout(rect=[0,0,1,.95])
        fig.savefig(figs/f'{stem}.png',bbox_inches='tight');fig.savefig(figs/f'{stem}.pdf',bbox_inches='tight');plt.close(fig)
        index.append(dict(category=num,title=title,figure=f'figures/{stem}.png',vector=f'figures/{stem}.pdf',data=f'figures/data/{stem}.csv',scope='Scope A unless explicitly labeled',caption=title+'; chronological holdout, project-defined states.'))
    def bars(num,title,data,cols):
        fig,ax=plt.subplots(figsize=(11,5));x=np.arange(len(data));width=.8/len(cols)
        for j,col in enumerate(cols):
            vals=data[col].to_numpy(float);rect=ax.bar(x+(j-(len(cols)-1)/2)*width,vals,width,label=col.replace('_',' '))
            artist_checks.append(dict(category=num,series=col,max_error=float(max(abs(a.get_height()-b) for a,b in zip(rect,vals)))))
        ax.set_xticks(x,data.model,rotation=35,ha='right');ax.set_ylim(0,1.05);ax.set_ylabel('Fraction');ax.legend(fontsize=9)
        save(num,title,data,fig)
    main=test[test.model.isin(['E0','E1','E2',selected])]
    bars(1,'Accuracy and macro F1',main,['accuracy','macro_f1'])
    bars(2,'Change recall across models',test,['change_recall'])
    bars(3,'Correct destination on actual transitions',test,['exact_destination'])
    fig,ax=plt.subplots(figsize=(8,5));ax.scatter(test.stable_accuracy,test.change_recall,s=70)
    for _,r in test.iterrows():ax.annotate(r.model,(r.stable_accuracy,r.change_recall),xytext=(5,4),textcoords='offset points')
    ax.set(xlabel='Stable-state accuracy',ylabel='Change recall',xlim=(.78,1.02),ylim=(-.03,.8));save(4,'Stability versus transition recall',test,fig)
    bars(5,'Transition precision recall and F1',main,['change_precision','change_recall','change_f1'])
    diff=pd.read_csv(root/'results/model_improvement_vs_baselines.csv').query("baseline=='E2'")
    ci=pd.read_csv(root/'results/bootstrap_confidence_intervals.csv').query("reference=='E2'")[['metric','lower','upper']]
    diff=diff.merge(ci,on='metric',validate='one_to_one')
    errors=np.stack([diff.percentage_points-100*diff.lower,100*diff.upper-diff.percentage_points])
    fig,ax=plt.subplots(figsize=(10,5));rect=ax.barh(diff.metric,diff.percentage_points,xerr=errors,capsize=3,color=np.where(diff.delta>=0,'#15847b','#b84a48'));ax.axvline(0,color='black',lw=.7);ax.set_xlabel('Selected E4 minus F3 (percentage points; paired 95% CI)')
    artist_checks.append(dict(category=6,series='percentage_points',max_error=float(max(abs(a.get_width()-b) for a,b in zip(rect,diff.percentage_points)))))
    save(6,'Selected Phase 7 versus Phase 6',diff,fig)
    feature=pd.read_csv(root/'results/missed_transition_analysis.csv');feature=feature[feature.current_state>=0].copy()
    feature['distance_bin']=pd.cut(feature.dmin,[0,5,10,20,40,80,np.inf],include_lowest=True)
    summary=feature.groupby('distance_bin',observed=True).agg(actual_changes=('actual_change','sum'),missed_F3=('missed_E2','sum'),missed_selected=(f'missed_{selected}','sum'),N=('y','size')).reset_index()
    fig,ax=plt.subplots(figsize=(9,5));x=np.arange(len(summary));ax.plot(x,summary.missed_F3/summary.actual_changes,'o-',label='F3');ax.plot(x,summary.missed_selected/summary.actual_changes,'o-',label=selected);ax.set_xticks(x,summary.distance_bin.astype(str),rotation=20);ax.set(xlabel='Distance to nearest threshold (ug/m3)',ylabel='Miss rate among actual transitions');ax.legend();save(7,'Threshold proximity and missed transitions',summary,fig)
    for num,cols,title in [(8,['v3','v6'],'Target station rates before transitions'),(9,['cross_mean','cross_max','fraction_rising'],'Other station rates before transitions')]:
        summary=feature.groupby('actual_change')[cols].agg(['mean','median','std']).reset_index();summary.columns=['_'.join(str(x) for x in c if str(x)) if isinstance(c,tuple) else c for c in summary.columns]
        fig,axes=plt.subplots(1,len(cols),figsize=(12,4))
        for ax,col in zip(axes,cols):
            ax.boxplot([feature.loc[~feature.actual_change,col],feature.loc[feature.actual_change,col]],tick_labels=['Stable','Change'],showfliers=False);ax.set_title(col);ax.set_ylabel('Fraction' if col=='fraction_rising' else 'ug/m3/hour')
        # Preserve exact observations used by boxplots as well as aggregate CSV.
        feature[['actual_change']+cols].to_csv(figs/f'data/{num:02d}_observations.csv',index=False)
        save(num,title,summary,fig)
    bars(10,'Controlled E0 to E7 ablations',test,['macro_f1','change_f1','J'])
    grid=pd.read_csv(root/'results/precursor_fusion_grid.csv');lock=json.loads((root/'config/selected_configuration.json').read_text())
    data=grid[grid.classifier.isin([lock['winners'][e]['classifier'] for e in ['E3','E4','E5']])]
    fig,ax=plt.subplots(figsize=(8,5))
    for name,g in data.groupby('group'):ax.plot(g.alpha,g.J,'o-',label=name)
    ax.set(xlabel='Alpha',ylabel='Validation Scope A J');ax.legend();save(11,'Validation fusion weight sensitivity',data,fig)
    data=pd.read_csv(root/'results/transition_threshold_grid.csv');fig,ax=plt.subplots(figsize=(8,5))
    for col in ['J','change_recall','change_precision','stable_accuracy']:ax.plot(data.theta,data[col],'o-',label=col)
    ax.set(xlabel='Theta',ylabel='Validation Scope A score');ax.legend();save(12,'Validation decision threshold sensitivity',data,fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4));data=test[test.model.isin(['E0','E2',selected])]
    for ax,(_,r) in zip(axes,data.iterrows()):
        cm=np.array([[r[f'cm_{i}{j}'] for j in range(3)] for i in range(3)]);ax.imshow(cm,cmap='Blues');ax.set_title(r.model);ax.set_xticks(range(3),['Low','Moderate','High'],rotation=20);ax.set_yticks(range(3),['Low','Moderate','High']);ax.set(xlabel='Predicted',ylabel='Actual')
        for i in range(3):
            for j in range(3):ax.text(j,i,str(int(cm[i,j])),ha='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
    save(13,'Three state confusion matrices',data,fig)
    rel=[];fig,axes=plt.subplots(1,2,figsize=(11,5))
    for name in ['E0','E2',selected]:
        f=frames[name];f=f[f.current_state>=0];p=f[['p0','p1','p2']].to_numpy()
        for kind,confidence,outcome in [('class',p.max(1),p.argmax(1)==f.y.to_numpy()),('change',f.q_new.to_numpy(),f.y.to_numpy()!=f.current_state.to_numpy())]:
            bins=np.minimum((confidence*10).astype(int),9)
            for b in range(10):
                mask=bins==b
                if mask.any():rel.append(dict(model=name,kind=kind,bin=b,N=int(mask.sum()),predicted=float(confidence[mask].mean()),observed=float(outcome[mask].mean())))
    rel=pd.DataFrame(rel)
    for ax,kind in zip(axes,['class','change']):
        ax.plot([0,1],[0,1],'--',color='gray')
        for name,g in rel[rel.kind==kind].groupby('model'):ax.plot(g.predicted,g.observed,'o-',label=name)
        ax.set(xlabel='Mean predicted probability',ylabel='Observed frequency',title=kind);ax.legend()
    save(14,'Probability reliability',rel,fig)
    costs=pd.read_csv(root/'results/computational_cost.csv');rows=[]
    for name,w in lock['winners'].items():
        c=costs[costs.model==w['classifier']].iloc[0];m=test[test.model==name].iloc[0];rows.append(dict(model=name,fit_seconds=c.fit_seconds,model_bytes=c.model_bytes,prediction_seconds=m.prediction_seconds,J=m.J))
    costs=pd.DataFrame(rows);fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].scatter(costs.prediction_seconds,costs.J);axes[0].set(xlabel='Warm prediction seconds (5,720 rows)',ylabel='Test Scope A J')
    axes[1].bar(costs.model,costs.model_bytes/1024);axes[1].set_ylabel('New classifier artifact (KiB)')
    for _,r in costs.iterrows():axes[0].annotate(r.model,(r.prediction_seconds,r.J))
    save(15,'Additional model computation and storage',costs,fig)
    pca=pd.read_csv(repo/'phase4_output/pca_hosvd_comparison.csv');data=pca[pca.split=='test'];fig,ax=plt.subplots(figsize=(8,5))
    for name,g in data.groupby('method'):g=g.sort_values('k');ax.plot(g.k,g.relative_reconstruction_error,'o-',label=name)
    ax.set(xlabel='Compressed dimension',ylabel='Relative reconstruction error');ax.legend();save(16,'PCA and HoSVD reconstruction',data,fig)
    data=pd.read_csv(repo/'Phase5_SVM_Kernel_RFF_Results/results/stage_a_best_per_config.csv');dim=pca[pca.split=='train'].copy();dim['config']=dim.method+'_'+dim.config_id
    data=data.merge(dim[['config','k','r_time','r_station','r_feature']],on='config',validate='one_to_one');fig,ax=plt.subplots(figsize=(8,5))
    for name,g in data.groupby(data.config.str.split('_').str[0]):ax.plot(g.k,g.val_scopeA_macro_f1,'o-',label=name)
    ax.set(xlabel='Compressed dimension',ylabel='Validation Scope A macro-F1 (linear SVM)');ax.legend();save(17,'Compression dimension and classification',data,fig)
    data=cross[cross.model.str.contains('stageB_full|revised_rff',regex=True,na=False)].copy();data.model=data.model.str.replace('final_','').str.replace('stageB_full','RBF').str.replace('revised_rff','RFF')
    bars(18,'Exact RBF versus revised RFF',data,['accuracy','macro_f1','change_f1'])
    bars(19,'Transitions into High pollution',main,['high_precision','high_recall','high_f1'])
    order=['E0','final_linear_best_overall','final_stageB_full_pca_B','final_revised_rff_pca_B','Phase6_f1','Phase6_f2','E2',selected]
    data=cross.set_index('model').loc[order].reset_index().copy();data.model=data.model.str.replace('final_','')
    bars(20,'Research progression on matching test rows',data,['accuracy','change_f1'])
    pd.DataFrame(index).to_csv(figs/'figure_index.csv',index=False)
    pd.DataFrame(artist_checks).to_csv(figs/'artist_value_checks.csv',index=False)
    return cross
