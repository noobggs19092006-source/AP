"""Calibration diagnostics only; no post-test calibrator is fitted."""
import numpy as np
def confidence_ece(p,y,n_bins=10):
    confidence=p.max(1);correct=p.argmax(1)==y;bins=np.minimum((confidence*n_bins).astype(int),n_bins-1)
    return float(sum(abs(correct[bins==k].mean()-confidence[bins==k].mean())*(bins==k).mean() for k in range(n_bins) if (bins==k).any()))
