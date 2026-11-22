"""Explicit logistic equation, independent of sklearn predict_proba."""
import numpy as np
def logistic_from_parameters(features,pipeline):
    z=(features-pipeline[0].mean_)/pipeline[0].scale_
    return 1/(1+np.exp(-(z@pipeline[1].coef_[0]+pipeline[1].intercept_[0])))
