"""Pair-preserving percentile contrast from independently resampled metrics."""
import numpy as np
def percentile_contrast(samples,model,reference,column):
    values=np.asarray(samples[model])[:,column]
    if reference!='absolute':values=values-np.asarray(samples[reference])[:,column]
    return np.quantile(values,[.025,.975])
