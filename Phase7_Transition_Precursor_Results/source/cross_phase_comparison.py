"""Enhance recomputed cross-phase table with explicit representation provenance."""
import numpy as np
import pandas as pd
def annotate(table,repo,root):
    table=table.copy();table['representation']='N/A';table['feature_dimension']=np.nan;table['model_storage_bytes']=np.nan
    dimensions={'A':48,'B':192,'C':384}
    # Read actual feature shapes instead of assuming ranks from names.
    for i,r in table.iterrows():
        name=str(r.model)
        if name=='E0':table.loc[i,'representation']='observed current state';table.loc[i,'feature_dimension']=1
        elif name in ['E1','E2']:
            table.loc[i,'representation']='pca_C';table.loc[i,'feature_dimension']=np.load(repo/'phase4_output/features_test_pca_C.npz')['F'].shape[1]
        elif name.startswith('E'):
            table.loc[i,'representation']='original PCA-C SVM plus precursor features'
            table.loc[i,'feature_dimension']={'E1':0,'E2':0,'E3':6,'E4':8,'E5':11,'E6':11,'E7':11}.get(name,np.nan)
        else:
            cfg=next((f'{method}_{level}' for method in ['pca','hosvd'] for level in ['A','B','C'] if f'{method}_{level}' in name),None)
            cfg={'Phase6_f1':'hosvd_B','Phase6_f2':'pca_B','Phase6_f3':'pca_C'}.get(name,cfg)
            if 'linear_best_overall' in name:cfg='pca_C'
            if cfg and (repo/f'phase4_output/features_test_{cfg}.npz').exists():
                table.loc[i,'representation']=cfg;table.loc[i,'feature_dimension']=np.load(repo/f'phase4_output/features_test_{cfg}.npz')['F'].shape[1]
            model=repo/f'Phase5_SVM_Kernel_RFF_Results/models/{name}.pkl'
            if model.exists():table.loc[i,'model_storage_bytes']=model.stat().st_size
    table['dimension_note']='For new E3-E7, dimension is additional precursor features; SVM representation is reused. RFF dimension column describes compressed input, not random map output.'
    return table
