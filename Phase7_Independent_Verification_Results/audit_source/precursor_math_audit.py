"""Alternative feature equations used only by the independent audit."""
import numpy as np
def independent_features(physical,state):
    cur=physical[:,-1,0];other=np.diff(physical[:,[20,23],1:],axis=1)[:,0,:]/3
    return np.column_stack((abs(cur-35),abs(cur-75),np.minimum(abs(cur-35),abs(cur-75)),state==0,state==1,state==2,(cur-physical[:,20,0])/3,(cur-physical[:,17,0])/6,other.mean(1),other.max(1),(other>0).mean(1)))
