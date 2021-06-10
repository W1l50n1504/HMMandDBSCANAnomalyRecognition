import numpy as np
import seaborn as sns
import numpy as np
import pandas as pd
import os

from hmmlearn.hmm import GaussianHMM
from matplotlib import pyplot as plt
from scipy import stats as ss
from keras import *
from utility import *


def fitHMM():
    n_components = 6

    X_train, y_train, X_test, y_test = loadDataHMM()

    X_train, y_train, X_test, y_test, X_val, y_val = dataProcessingHMM(X_train, y_train, X_test, y_test)

    print('Preparazione matrici di probabilità...')
    startprob = np.zeros(n_components)
    startprob[0] = 1

    transmat = np.zeros((n_components, n_components))
    transmat[0, 0] = 1
    transmat[-1, -1] = 1

    for i in range(transmat.shape[0] - 1):
        if i != transmat.shape[0]:
            for j in range(i, i + 2):
                transmat[i, j] = 0.5

    print('Creazione modello...')
    # model = GaussianHMM(n_components=n_components, covariance_type="diag", init_params="cm", params="cm")
    # Mettere verbose=True se si vuole vedere la creazione delle prob
    model = GaussianHMM(n_components=3, n_iter=1000, verbose=False, init_params="cm")

    model.startprob_ = np.array(startprob)
    model.transmat_ = np.array(transmat)

    print('Fitting del modello...')
    model.fit(X_train, y_train);
    print('Plotting dei risultati del modello')
    train_scores = []
    test_scores = []
    val_scores = []

    for i in range(len(np.array(y_train))):
        train_score = model.score(X_train[X_train == i])
        train_scores.append(train_score)

    for i in range(len(np.array(y_test))):
        test_score = model.score(X_test[X_test == i])
        test_scores.append(test_score)

    for i in range(len(np.array(y_val))):
        val_score = model.score(X_val[X_val == i])
        val_scores.append(val_score)

    length_train = len(train_scores)
    length_val = len(val_scores) + length_train
    length_test = len(test_scores) + length_val

    plt.figure(figsize=(7, 5))
    plt.scatter(np.arange(length_train), train_scores, c='b', label='trainset')
    plt.scatter(np.arange(length_train, length_val), val_scores, c='r', label='testset - imitation')
    plt.scatter(np.arange(length_val, length_test), test_scores, c='g', label='testset - original')
    plt.title(f'User: 1 | HMM states: {n_components} | GMM components: 2')
    plt.legend(loc='lower right')

    plt.savefig(hmmGraph)
    plt.show()


if __name__ == '__main__':
    """
    #effettuo fitting del modello
    hidden_states, mus, sigmas, P, samples = fit_hmm(X_train)

    #plot del modello
    plotTimeSeries(X_train, hidden_states)
    plotDistribution(X_train, mus, sigmas, P)

    #length_train, length_val, length_test, train_scores, test_scores, val_scores = fit_hmm()
    #PlotHMM(length_train, length_val, length_test, train_scores, test_scores, val_scores)
    """
    fitHMM()
