"""F3 adapter and precursor fusion. Unknown states always use feature-only SVM."""
import numpy as np
from .transition_decision import hard_decision

def f3(s,svm,P):
    valid=s>=0; ids=np.flatnonzero(valid); cs=s[valid]
    markov=svm.copy(); markov[valid]=P[cs]
    q=np.full(len(s),np.nan); d=np.zeros_like(svm)
    q[valid]=.75*(1-svm[ids,cs])+.25*(1-markov[ids,cs])
    d[valid]=markov[valid]; d[ids,cs]=0
    denom=d[valid].sum(1); d[valid]/=denom[:,None]
    return markov,q,d,reconstruct(q,d,s,svm)

def reconstruct(q,d,s,fallback):
    p=fallback.copy(); ids=np.flatnonzero(s>=0)
    p[ids]=q[ids,None]*d[ids]; p[ids,s[ids]]=1-q[ids]
    assert np.isfinite(p).all() and np.allclose(p.sum(1),1) and p.min()>=0
    return p

def predict(qp,qf,d,s,svm,alpha,theta=None):
    q=alpha*qp+(1-alpha)*qf
    p=reconstruct(q,d,s,svm); pred=hard_decision(p,q,d,s,theta)
    return p,pred,q
