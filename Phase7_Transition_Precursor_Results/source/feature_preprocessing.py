"""Training-only scaling primitive used by the logistic pipeline."""
from sklearn.preprocessing import StandardScaler
def new_scaler():return StandardScaler()
