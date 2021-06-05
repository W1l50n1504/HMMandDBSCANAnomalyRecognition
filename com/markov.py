from hmmlearn.hmm import GaussianHMM
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats as ss
import numpy as np
import pandas as pd
from keras import *
import os

from utility import *

trainset = '/dataset/trainset.csv'
testset = '/dataset/testset.csv'

absPath_ = os.getcwd()
pathTrain = absPath_ + trainset
pathTest = absPath_ + testset

mydpi = 96


def fit(logdata):

    # effettua il fitting dei modelli sui dati caricati dal dataset
    print('Creazione del modello e fitting dei dati...')
    model = GaussianHMM(n_components=6, n_iter=1000).fit(np.reshape(logdata, [len(logdata), 1]))
    print('fine fitting')

    # classificazione di ogni osservazione come stato
    print('creazione hidden states')
    hidden_states = model.predict(np.reshape(logdata, [len(logdata), 1]))
    print('fine creazione hidden states')

    # trova i parametri di un HMM Gaussiano
    mus = np.array(model.means_)
    sigmas = np.array(np.sqrt(np.array([np.diag(model.covars_[0]), np.diag(model.covars_[1])])))
    P = np.array(model.transmat_)

    # trova la log-likelihood di una HMM Gaussiana
    logProb = model.score(np.reshape(data, [len(logdata), 1]))

    # genera nSamples dagli HMM Gaussiani
    samples = model.sample(nSamples)

    # riorganizza i mu, i sigma e P in modo che la prima colonna contenga i lower mean (se non già presenti)
    if mus[0] > mus[1]:
        mus = np.flipud(mus)
        sigmas = np.flipud(sigmas)
        P = np.fliplr(np.flipud(P))
        hidden_states = 1 - hidden_states


if __name__ == '__main__':
