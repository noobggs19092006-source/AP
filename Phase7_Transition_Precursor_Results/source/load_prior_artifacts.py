"""Load trusted local saved model, original calibration and representation scaler."""
import pickle
import numpy as np
def load_linear(repo):
    folder=repo/'Phase5_SVM_Kernel_RFF_Results/models'
    model=pickle.load(open(folder/'final_linear_best_overall.pkl','rb'))
    calibrator=pickle.load(open(folder/'calibrator_linear_best_overall.pkl','rb'))
    scaler=np.load(folder/'scaler_pca_C.npz')
    assert list(model.classes_)==list(calibrator.classes_)==[0,1,2]
    return model,calibrator,scaler
