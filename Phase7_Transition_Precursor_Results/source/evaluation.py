"""Scope-explicit metrics. Zero-denominator event scores are zero."""
import numpy as np
from sklearn.metrics import confusion_matrix
from .probability_calibration import confidence_ece

def ratio(a,b): return float(a/b) if b else 0.

def metrics(y,pred,s,p=None,scope='A'):
    take=s>=0 if scope=='A' else np.ones(len(s),dtype=bool)
    y,pred,s=y[take],pred[take],s[take]
    cm=confusion_matrix(y,pred,labels=[0,1,2]); tp=np.diag(cm)
    precision=np.divide(tp,cm.sum(0),out=np.zeros(3),where=cm.sum(0)>0)
    recall=np.divide(tp,cm.sum(1),out=np.zeros(3),where=cm.sum(1)>0)
    f1=np.divide(2*tp,cm.sum(0)+cm.sum(1),out=np.zeros(3),where=(cm.sum(0)+cm.sum(1))>0)
    out={'N':len(y),'accuracy':float((y==pred).mean()),'macro_f1':float(f1.mean())}
    for k in range(3):
        out.update({f'class{k}_precision':precision[k],f'class{k}_recall':recall[k],f'class{k}_f1':f1[k]})
        for j in range(3): out[f'cm_{k}{j}']=int(cm[k,j])
    valid=s>=0; a=(y!=s)&valid; b=(pred!=s)&valid
    TP=int((a&b).sum()); FP=int((~a&b&valid).sum()); FN=int((a&~b).sum()); TN=int((~a&~b&valid).sum())
    out.update(TP=TP,FP=FP,FN=FN,TN=TN,actual_changes=int(a.sum()),transition_N=int(valid.sum()),
               change_precision=ratio(TP,TP+FP),change_recall=ratio(TP,TP+FN),change_f1=ratio(2*TP,2*TP+FP+FN),
               exact_destination=ratio(((y==pred)&a).sum(),a.sum()),stable_accuracy=ratio(((y==pred)&~a&valid).sum(),(~a&valid).sum()))
    a=(y==2)&(s!=2)&valid; b=(pred==2)&(s!=2)&valid
    ht=int((a&b).sum()); hf=int((~a&b).sum()); hn=int((a&~b).sum())
    out.update(high_TP=ht,high_FP=hf,high_FN=hn,high_precision=ratio(ht,ht+hf),high_recall=ratio(ht,ht+hn),high_f1=ratio(2*ht,2*ht+hf+hn))
    if p is not None:
        p=p[take]; assert np.isfinite(p).all() and np.allclose(p.sum(1),1) and p.min()>=0
        clipped=np.clip(p,1e-12,1); clipped/=clipped.sum(1,keepdims=True)
        confidence=p.max(1); correct=p.argmax(1)==y; bins=np.minimum((confidence*10).astype(int),9)
        ece=confidence_ece(p,y)
        out.update(brier=float(np.square(p-np.eye(3)[y]).sum(1).mean()),log_loss=float(-np.log(clipped[np.arange(len(y)),y]).mean()),ece=float(ece))
    out['J']=.5*(out['macro_f1']+out['change_f1'])
    return out
