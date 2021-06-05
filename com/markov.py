from hmmlearn import hmm
from utility import *


def fit_hmm(n_mix=32):
    X_train, y_train, X_test, y_test = loadDataHMM()
    X_train, y_train, X_test, y_test, X_val, y_val = dataProcessingHMM(X_train, y_train, X_test, y_test)

    n_components = 6

    startprob = np.zeros(n_components)
    startprob[0] = 1

    transmat = np.zeros((n_components, n_components))
    transmat[0, 0] = 1
    transmat[-1, -1] = 1

    for i in range(transmat.shape[0] - 1):
        if i != transmat.shape[0]:
            for j in range(i, i + 2):
                transmat[i, j] = 0.5

    lr = hmm.GMMHMM(n_components=n_components,
                    n_mix=n_mix,
                    covariance_type="diag",
                    init_params="cm", params="cm")

    lr.startprob_ = np.array(startprob)
    lr.transmat_ = np.array(transmat)

    lr.fit(X_train, y_train);

    train_scores = []
    test_scores = []
    val_scores = []

    for i in range(len(np.array(y_train))):
        train_score = lr.score(X_train[i])
        train_scores.append(train_score)

    for i in range(len(np.array(y_test))):
        test_score = lr.score(X_test[i])
        test_scores.append(test_score)

    for i in range(len(np.array(y_val))):
        val_score = lr.score(X_val[i])
        val_scores.append(val_score)

        length_train = len(train_scores)
        length_val = len(val_scores) + length_train
        length_test = len(test_scores) + length_val


if __name__ == '__main__':
   # for j in range(1, 32):
      #  fit_hmm(n_mix=j)
    fit_hmm()