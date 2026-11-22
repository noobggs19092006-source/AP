"""Threshold labels do not alter the reported probability vector."""
import numpy as np
def hard_decision(p,q,d,s,theta=None):
    pred=p.argmax(1)
    if theta is not None:
        v=s>=0;pred[v]=np.where(q[v]>=theta,d[v].argmax(1),s[v])
    return pred
