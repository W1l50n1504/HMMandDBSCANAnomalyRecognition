"""
File contenente la classe riguardante il sistema degli Hidden Markov Models
n_components=6 perché vogliamo che il sistema sia in grado di riconoscere 6 tipi di attività diversi (per ora)
questi tipi di attività sono: scendere le scale, salire le scale, stare seduti, stare in piedi, camminare, correre
"""

import numpy as np
from hmmlearn.hmm import GaussianHMM
from matplotlib import pyplot as plt
import seaborn as sns

from com.utility import *


class HiddenMarkovModels():

    def __init__(self, data, nSamples):
        # in questa maniera abbiamo deciso il tipo di hmm che si vuole utilizzare e il numero di attività che si vuole riconoscere
        # e il numero di iterazioni che deve effettuare il sistema
        # data e' la lista contente i dati su cui allenare la rete

        self.data = data

        # fitting dei dati nel modello
        print('Creazione del modello e fitting dei dati...')
        self.model = GaussianHMM(n_components=6, n_iter=1000).fit(np.reshape(self.data, np.shape(self.data)))
        print('fine fitting')

        # classifica ogni osservazione c
        print('creazione hidden states')
        self.hidden_states = self.model.predict(np.reshape(self.data, np.shape(self.data)))
        print('fine creazione hidden states')

        # find parameters of Gaussian HMM
        self.mus = np.array(self.model.means_)
        self.sigmas = np.array(np.sqrt(np.array([np.diag(self.model.covars_[0]), np.diag(self.model.covars_[1])])))
        self.P = np.array(self.model.transmat_)

        # find log-likelihood of Gaussian HMM
        self.logProb = self.model.score(np.reshape(self.data, np.shape(self.data)))

        # generate nSamples from Gaussian HMM
        samples = self.model.sample(nSamples)

        # re-organize mus, sigmas and P so that first row is lower mean (if not already)
        if self.mus.any():
            # if self.mus[0] > self.mus[1]:
            self.mus = np.flipud(self.mus)
            self.sigmas = np.flipud(self.sigmas)
            self.P = np.fliplr(np.flipud(self.P))
            self.hidden_states = 1 - self.hidden_states

    def plot(self, ylabel, filename):
        print('Inizio plotting degli HMM')

        sns.set()
        fig = plt.figure()
        ax = fig.add_subplot(111)

        xs = np.arange(len(self.data))
        #print(len(self.data)) = 7352


        masks = self.hidden_states == 0 #7352
        print('xs[masks]', len(xs[masks]))
        print('self.data[masks]', len(self.data[masks]))
        ax.scatter(xs[masks], self.data[masks], c='r', label='STANDING')


        masks = self.hidden_states == 1
        ax.scatter(xs[masks], self.data[masks], c='b', label='SITTING')

        masks = self.hidden_states == 2
        ax.scatter(xs[masks], self.data[masks], c='g', label='LAYING')

        masks = self.hidden_states == 3
        ax.scatter(xs[masks], self.data[masks], c='y', label='WALKING')

        masks = self.hidden_states == 4
        ax.scatter(xs[masks], self.data[masks], c='c', label='WALKING DOWNSTAIRS')

        masks = self.hidden_states == 5
        ax.scatter(xs[masks], self.data[masks], c='w', label='WALKING UPSTAIRS')

        ax.plot(xs, self.data, c='k')

        ax.set_xlabel('Year')
        ax.set_ylabel(ylabel)
        fig.subplots_adjust(bottom=0.2)
        handles, labels = plt.gca().get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, frameon=True)
        fig.savefig(filename)
        fig.clf()

        return None


if __name__ == '__main__':
    ds = Dataset()
    trainData_X, trainActivity_y, testData_X, testActivity_y = ds.main()

    ylabel = 'ylabel'
    filename = absPath_ + '/immagine/grafico.png'

    # print('\ntrainData_X\n', trainData_X)

    # print('\ntrainActivity_y\n', trainActivity_y)

    # print('\ntestData_X\n', testData_X)

    # print('\ntestActivity_y\n', testActivity_y)

    # print(np.reshape(trainData_X, np.shape(trainData_X)), 6)

    hmm = HiddenMarkovModels(trainData_X, 100)

    plt.switch_backend('agg')  # turn off display when running with Cygwin
    hmm.plot(ylabel, filename)
