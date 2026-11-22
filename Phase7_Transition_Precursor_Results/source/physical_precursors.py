"""Verified hourly, station-specific precursor extraction; legacy module is unused."""
import numpy as np

NAMES = ['d35','d75','dmin','state_low','state_moderate','state_high',
         'v3','v6','cross_mean','cross_max','fraction_rising']
GROUPS = {'threshold': list(range(6)), 'temporal': list(range(8)), 'all': list(range(11))}

def states(m):
    m=np.asarray(m)
    return np.where(np.isfinite(m),np.where(m<35,0,np.where(m<75,1,2)),-1)

def extract(X, mean, std, current):
    assert X.shape[1:]==(24,12,11)
    assert mean.shape==std.shape==(12,11)
    # Float32 standardized histories introduce small physical reconstruction errors.
    m=X[:,:,:,0].astype(float)*std[None,None,:,0]+mean[None,None,:,0]
    now=m[:,23,0]; d35=abs(now-35); d75=abs(now-75)
    other=(m[:,23,1:]-m[:,20,1:])/3
    h=np.column_stack([d35,d75,np.minimum(d35,d75),current==0,current==1,current==2,
                       (now-m[:,20,0])/3,(now-m[:,17,0])/6,
                       other.mean(1),other.max(1),(other>0).mean(1)])
    assert np.isfinite(h).all()
    return h,m
