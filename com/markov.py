from hmmlearn import hmm
from utility import *


def fit_hmm():
    X_train, y_train, X_test, y_test = loadDataHMM()
    X_train, y_train, X_test, y_test, X_val, y_val = dataProcessingHMM(X_train, y_train, X_test, y_test)

    print('Creazione del modello...')
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

    model = hmm.GMMHMM(n_components=n_components, covariance_type="diag")

    model.startprob_ = np.array(startprob)
    model.transmat_ = np.array(transmat)

    print('Fine creazione del modello')

    print('Inizio fitting del modello...')
    # print('X_train\n', X_train)
    model.fit(X_train);
    print('Fine fitting del modello')

    print('Inizio valutazione...')
    train_scores = []
    test_scores = []
    val_scores = []

    print('prima parte')
    for i in range(len(np.array(y_train))):
        train_score = model.score(X_train)
        train_scores.append(train_score)

    print('seconda parte')
    for i in range(len(np.array(y_test))):
        test_score = model.score(X_test)
        test_scores.append(test_score)

    print('terza parte')
    for i in range(len(np.array(y_val))):
        val_score = model.score(X_val)
        val_scores.append(val_score)

        length_train = len(train_scores)
        length_val = len(val_scores) + length_train
        length_test = len(test_scores) + length_val
    print('Fine valutazione')

    return length_train, length_val, length_test


if __name__ == '__main__':
    length_train, length_val, length_test = fit_hmm()
    PlotHMM(length_train, length_val, length_test)
