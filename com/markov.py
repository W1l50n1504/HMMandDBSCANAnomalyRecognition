"""
File contenente la classe riguardante il sistema degli Hidden Markov Models
n_components=6 perché vogliamo che il sistema sia in grado di riconoscere 6 tipi di attività diversi (per ora)
questi tipi di attività sono: scendere le scale, salire le scale, stare seduti, stare in piedi, camminare, correre
"""

import numpy as np
from hmmlearn.hmm import GaussianHMM
from com.utility import *


class HiddenMarkovModels():

    def __init__(self, data, nSamples):
        # in questa maniera abbiamo deciso il tipo di hmm che si vuole utilizzare e il numero di attività che si vuole riconoscere
        # e il numero di iterazioni che deve effettuare il sistema
        # data e' la lista contente i dati su cui allenare la rete

        self.model = GaussianHMM(n_components=6, n_iter=1000).fit(np.reshape(data, [len(data), 1]))

        # classifica ogni osservazione c
        self.hidden_states = self.model.predict(np.reshape(data, [len(data), 1]))

    def fitHMM(self, data, nSamples):
        # fit Gaussian HMM to Q
        model = GaussianHMM(n_components=2, n_iter=1000).fit(np.reshape(data, [len(data), 1]))

        # classify each observation as state 0 or 1
        hidden_states = model.predict(np.reshape(data, [len(data), 1]))

        # find parameters of Gaussian HMM
        mus = np.array(model.means_)
        sigmas = np.array(np.sqrt(np.array([np.diag(model.covars_[0]), np.diag(model.covars_[1])])))
        P = np.array(model.transmat_)

        # find log-likelihood of Gaussian HMM
        logProb = model.score(np.reshape(data, [len(data), 1]))

        # generate nSamples from Gaussian HMM
        samples = model.sample(nSamples)

        # re-organize mus, sigmas and P so that first row is lower mean (if not already)
        if mus[0] > mus[1]:
            mus = np.flipud(mus)
            sigmas = np.flipud(sigmas)
            P = np.fliplr(np.flipud(P))
            hidden_states = 1 - hidden_states

        return hidden_states, mus, sigmas, P, logProb, samples

    # load annual flow data for the Colorado River near the Colorado/Utah state line
    # AnnualQ = np.loadtxt('AnnualQ.txt')

    # log transform the data and fit the HMM
    # logQ = np.log(AnnualQ)
    # hidden_states, mus, sigmas, P, logProb, samples = fitHMM(logQ, 100)


if __name__ == '__main__':
    ds = Dataset()
    ds.main(filename_)
    accel = ds.retAccel()

    print(accel)
