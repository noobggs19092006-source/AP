"""Independent baseline reconstruction using original model and matrix."""
import pickle
import numpy as np
import pandas as pd

def verify(root,repo):
    P=pd.read_csv(repo/'Air_Pollution_Phase3_Results/phase3_output/markov_transition_matrix.csv').iloc[:,1:].to_numpy(float)
    modeldir=repo/'Phase5_SVM_Kernel_RFF_Results/models';model=pickle.load(open(modeldir/'final_linear_best_overall.pkl','rb'));cal=pickle.load(open(modeldir/'calibrator_linear_best_overall.pkl','rb'));sc=np.load(modeldir/'scaler_pca_C.npz');rows=[]
    for split in ['validation','test']:
        e1=pd.read_csv(root/f'predictions/{split}/E1.csv');e2=pd.read_csv(root/f'predictions/{split}/E2.csv');feat=np.load(repo/f'phase4_output/features_{split}_pca_C.npz')
        scores=model.decision_function((feat['F']-sc['mean'])/sc['scale']);svm=cal.predict_proba(scores)
        error_svm=np.max(abs(svm-e1[['p0','p1','p2']].to_numpy()))
        s=e2.current_state.to_numpy();v=s>=0;rows_id=np.flatnonzero(v);q=.75*(1-svm[rows_id,s[v]])+.25*(1-P[s[v],s[v]])
        expected=np.zeros((sum(v),3))
        for k,i in enumerate(s[v]):
            for j in range(3):expected[k,j]=(1-q[k]) if i==j else q[k]*P[i,j]/(1-P[i,i])
        error_f3=np.max(abs(expected-e2.loc[v,['p0','p1','p2']].to_numpy()))
        rows.append(dict(split=split,svm_max_error=error_svm,f3_max_error=error_f3,status='PASS' if max(error_svm,error_f3)<1e-8 else 'FAIL'))
    return pd.DataFrame(rows)
