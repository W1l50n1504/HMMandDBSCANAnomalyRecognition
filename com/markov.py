"""
File contenente la classe riguardante il sistema degli Hidden Markov Models
"""
import numpy as np
from hmmlearn import hmm

np.random.seed(42)


class HiddenMarkovModels():

    def __init__(self):
        #
        self.model = hmm.GaussianHMM(n_components=3, covariance_type="full")
        #
        self.model.startprob_ = np.array([0.6, 0.3, 0.1])
        #
        self.model.transmat_ = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2], [0.3, 0.3, 0.4]])
        #
        self.model.means_ = np.array([[0.0, 0.0], [3.0, -3.0], [5.0, 10.0]])
        #
        self.model.covars_ = np.tile(np.identity(2), (3, 1, 1))
        #
        self.X, self.Z = self.model.sample(100)
        print('HMMs creati')

    def nuovaFunzione(self):
        return "ciao"


if __name__ == '__main__':
    hmm1 = HiddenMarkovModels()
    print(hmm1.nuovaFunzione())
