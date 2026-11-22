"""Reproducible regularized binary classifier construction."""
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from .feature_preprocessing import new_scaler
def new_classifier(C,weight):
    return make_pipeline(new_scaler(),LogisticRegression(C=C,class_weight=weight,max_iter=2000,random_state=42,tol=1e-9))
