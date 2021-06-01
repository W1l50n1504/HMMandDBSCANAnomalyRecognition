from hmmlearn.hmm import GaussianHMM
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats as ss

from utility import *

ylabel = 'Indice'
filename1 = absPath_ + '/immagine/grafico.png'
filename2 = absPath_ + '/immagine/graficoDistribuzione.png'

mydpi = 96


class HiddeMarkovModels:

    def __init__(self, nSamples):

        self.ds = Dataset(walkDown)
        self.ds.main()
        self.data = self.ds.toList()
        self.logdata = np.log(self.data)

        #effettua il fitting dei modelli sui dati caricati dal dataset
        print('Creazione del modello e fitting dei dati...')
        self.model = GaussianHMM(n_components=2, n_iter=1000).fit(np.reshape(self.logdata, [len(self.logdata), 1]))
        print('fine fitting')
        # classificazione di ogni osservazione come stato 1 o 2 (al momento riconosce solo un tipo di attività)
        print('creazione hidden states')
        self.hidden_states = self.model.predict(np.reshape(self.logdata, [len(self.logdata), 1]))
        print('fine creazione hidden states')
        # trova i parametri di un HMM Gaussiano
        self.mus = np.array(self.model.means_)
        self.sigmas = np.array(np.sqrt(np.array([np.diag(self.model.covars_[0]), np.diag(self.model.covars_[1])])))
        self.P = np.array(self.model.transmat_)

        # trova la log-likelihood di una HMM Gaussiana
        self.logProb = self.model.score(np.reshape(self.data, [len(self.data), 1]))

        # genera nSamples dagli HMM Gaussiani
        self.samples = self.model.sample(nSamples)

        # riorganizza i mu, i sigma e P in modo che la prima colonna contenga i lower mean (se non già presenti)
        if self.mus[0] > self.mus[1]:
            self.mus = np.flipud(self.mus)
            self.sigmas = np.flipud(self.sigmas)
            self.P = np.fliplr(np.flipud(self.P))
            self.hidden_states = 1 - self.hidden_states

    def plotScatter(self):
        sns.set()
        fig = plt.figure()
        ax = fig.add_subplot(111)

        xs = np.arange(len(self.logdata))

        masks = self.hidden_states == 0
        ax.scatter(xs[masks], self.logdata[masks], c='r', label='WalkingDwnStairs')

        masks = self.hidden_states == 1
        ax.scatter(xs[masks], self.logdata[masks], c='b', label='NotWalkingDwnStairs')
        #decommentare per congiungere tutti i punti sul grafico
        # ax.plot(xs, self.logdata, c='k')

        ax.set_xlabel('Valore')
        ax.set_ylabel(ylabel)
        fig.subplots_adjust(bottom=0.2)
        handles, labels = plt.gca().get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, frameon=True)
        fig.set_size_inches(800 / mydpi, 800 / mydpi)
        fig.savefig(filename1)
        fig.clf()

    def plotDistribution(self):
        # calcolo della distribuzione stazionaria
        eigenvals, eigenvecs = np.linalg.eig(np.transpose(self.P))
        one_eigval = np.argmin(np.abs(eigenvals - 1))
        pi = eigenvecs[:, one_eigval] / np.sum(eigenvecs[:, one_eigval])

        x_0 = np.linspace(self.mus[0] - 4 * self.sigmas[0], self.mus[0] + 4 * self.sigmas[0], 10000)
        fx_0 = pi[0] * ss.norm.pdf(x_0, self.mus[0], self.sigmas[0])

        x_1 = np.linspace(self.mus[1] - 4 * self.sigmas[1], self.mus[1] + 4 * self.sigmas[1], 10000)
        fx_1 = pi[1] * ss.norm.pdf(x_1, self.mus[1], self.sigmas[1])

        x = np.linspace(self.mus[0] - 4 * self.sigmas[0], self.mus[1] + 4 * self.sigmas[1], 10000)
        fx = pi[0] * ss.norm.pdf(x, self.mus[0], self.sigmas[0]) + \
             pi[1] * ss.norm.pdf(x, self.mus[1], self.sigmas[1])

        sns.set()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(self.logdata, color='k', alpha=0.5, density=True)
        l1, = ax.plot(x_0, fx_0, c='r', linewidth=2, label='WalkingDwnStairs Distn')
        l2, = ax.plot(x_1, fx_1, c='b', linewidth=2, label='NotWalkingDwnStairs Distn')
        l3, = ax.plot(x, fx, c='k', linewidth=2, label='Combined State Distn')

        fig.subplots_adjust(bottom=0.15)
        handles, labels = plt.gca().get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=3, frameon=True)
        fig.set_size_inches(800 / mydpi, 800 / mydpi)
        fig.savefig(filename2)
        fig.clf()


if __name__ == '__main__':
    hmm = HiddeMarkovModels(100)
    hmm.plotScatter()
    # hmm.plotDistribution()
