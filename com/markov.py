import numpy as np
import seaborn as sns
import numpy as np
import pandas as pd
import os
import pickle

from hmmlearn.hmm import GaussianHMM, GMMHMM
from matplotlib import pyplot as plt
from scipy import stats as ss
from keras import *
from utility import *

np.random.seed(42)

absPath_ = os.getcwd()
checkPointPathHMM = absPath_ + '/checkpoint/HMM'


def saveModel(model):
    with open(checkPointPathHMM + '/best_model.pkl', "wb") as file:
        pickle.dump(model, file)


def loadModel():
    with open(checkPointPathHMM + '/best_model.pkl', "rb") as file:
        model = pickle.load(file)

    return model


def fitHMM():
    n_mix = 16
    n_components = 6
    X_train, y_train, X_test, y_test = loadDataHMM()
    X_train, y_train, X_test, y_test, X_val, y_val = dataProcessingHMM(X_train, y_train, X_test, y_test)
    print('Creazioni matrici di prob...')
    startprob = np.zeros(n_components)
    startprob[0] = 1

    transmat = np.zeros((n_components, n_components))
    transmat[0, 0] = 1
    transmat[-1, -1] = 1

    for i in range(transmat.shape[0] - 1):
        if i != transmat.shape[0]:
            for j in range(i, i + 2):
                transmat[i, j] = 0.5
    print('Inizio fitting del modello...')
    lr = GMMHMM(n_components=n_components,
                n_mix=n_mix,
                covariance_type="diag",
                init_params="cm", params="cm", verbose=True)

    lr.startprob_ = np.array(startprob)
    lr.transmat_ = np.array(transmat)

    lr.fit(X_train);

    print('Salvataggio del modello...')
    saveModel(lr)


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = loadDataHMM()
    X_train, y_train, X_test, y_test, X_val, y_val = dataProcessingHMM(X_train, y_train, X_test, y_test)
    # fitHMM()
    plotHMM(X_train, y_train, X_test, y_test, X_val, y_val)
