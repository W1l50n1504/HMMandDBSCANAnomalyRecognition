from utility import *

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
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

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


def fitHMM(X_train, y_train, X_val, y_val):
    n_mix = 16
    n_components = 6
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
    model = GMMHMM(n_components=n_components,
                   n_mix=n_mix,
                   covariance_type="diag",
                   init_params="cm", params="cm", verbose=True)

    model.startprob_ = np.array(startprob)
    model.transmat_ = np.array(transmat)

    model.fit(X_train)

    print('Salvataggio del modello...')
    saveModel(model)


def matrixHMM(X_test, y_test, model):
    model = loadModel()
    rounded_labels = np.argmax(y_test, axis=1)
    y_pred = model.predict(X_test)


    print('rounded', rounded_labels.shape)
    print('y_pred', y_pred.shape)
    print('y_test', y_test.shape)

    print('y_test', y_test)
    print('rounded', rounded_labels)
    print('y_pred', y_pred)

    mat = confusion_matrix(rounded_labels, y_pred)
    plot_confusion_matrix(conf_mat=mat, show_normed=True, figsize=(10, 10))

    plt.figure(figsize=(10, 10))
    array = confusion_matrix(rounded_labels, y_pred)
    df_cm = pd.DataFrame(array, range(6), range(6))
    df_cm.columns = ["Walking", "W_Upstairs", "W_Downstairs", "Sitting", "Standing", "Laying"]
    df_cm.index = ["Walking", "W_Upstairs", "W_Downstairs", "Sitting", "Standing", "Laying"]
    # sn.set(font_scale=1)#for label size
    sns.heatmap(df_cm, annot=True, annot_kws={"size": 12},
                yticklabels=("Walking", "W_Upstairs", "W_Downstairs", "Sitting", "Standing", "Laying"),
                xticklabels=("Walking", "W_Upstairs", "W_Downstairs", "Sitting", "Standing", "Laying"))  # font size
    # plt.savefig()
    plt.show()


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = loadDataHMM()
    X_train, y_train, X_test, y_test, X_val, y_val = dataProcessingHMM(X_train, y_train, X_test, y_test)

    #fitHMM(X_train, y_train, X_val, y_val)
    model = loadModel()

    matrixHMM(X_test, y_test, model)
