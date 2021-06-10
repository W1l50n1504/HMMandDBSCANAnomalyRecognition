import matplotlib.pyplot as plt
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from scipy import stats as ss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import seaborn as sns
import numpy as np
import pandas as pd
import os

from cnn import *

absPath_ = os.getcwd()

X_train_signals_paths = absPath_ + '/dataset/train/X_train.txt'
X_test_signals_paths = absPath_ + '/dataset/test/X_test.txt'

y_train_path = absPath_ + '/dataset/train/y_train.txt'
y_test_path = absPath_ + '/dataset/test/y_test.txt'

pathToSignalTrain = absPath_ + '/dataset/train/Inertial Signals/'
pathToSignalTest = absPath_ + '/dataset/test/Inertial Signals/'

nameXtrain = 'total_acc_x_train.txt'
nameYtrain = 'total_acc_y_train.txt'
nameZtrain = 'total_acc_z_train.txt'

nameXtest = 'total_acc_x_test.txt'
nameYtest = 'total_acc_y_test.txt'
nameZtest = 'total_acc_z_test.txt'

hmmGraph = absPath_ + '/grafici/Markov/grafico.png'
hmmDistribution = absPath_ + '/grafici/Markov/distribution.png'

checkPointPathCNN = absPath_ + '/checkpoint/CNN'
checkPointPathBLSTM = absPath_ + '/checkpoint/BLSTM'

trainingValAccCNN = absPath_ + '/grafici/CNN/CNNAcc.png'
trainingValAccBLSTM = absPath_ + '/grafici/BLSTM/BLSTMAcc.png'

TrainingValAucCNN = absPath_ + '/grafici/CNN/CNNAuc.png'
TrainingValAucBLSTM = absPath_ + '/grafici/BLSTM/BLSTMAuc.png'

ModelLossCNN = absPath_ + '/grafici/CNN/ModelLossCNN.png'
ModelLossBLSTM = absPath_ + '/grafici/BLSTM/ModelLossBLSTM.png'

labelDict = {'WALKING': 0, 'WALKING_UPSTAIRS': 1, 'WALKING_DOWNSTAIRS': 2,
             'SITTING': 3, 'STANDING': 4, 'LAYING': 5}


def norm(data):
    return (data - data.mean()) / data.std() + np.finfo(np.float32).eps


def produceMagnitude(flag):
    magnitude = []
    if flag:

        x = norm(load_X(pathToSignalTrain + nameXtrain))
        y = norm(load_X(pathToSignalTrain + nameYtrain))
        z = norm(load_X(pathToSignalTrain + nameZtrain))

    else:
        x = norm(load_X(pathToSignalTest + nameXtest))
        y = norm(load_X(pathToSignalTest + nameYtest))
        z = norm(load_X(pathToSignalTest + nameZtest))

    for i in range(0, len(x)):
        magnitude.append(np.sqrt(x[i] ** 2 + y[i] ** 2 + z[i] ** 2))

    # print('\n', magnitude)

    return magnitude


def encode(train_X, train_y, test_X, test_y):
    # forse da eliminare
    train_y = train_y - 1
    test_y = test_y - 1
    # one hot encode y
    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)

    # print(train_X, train_y, test_X, test_y)

    return train_X, train_y, test_X, test_y


def load_X(X_signals_paths):
    X_signals = []

    file = open(X_signals_paths, 'r')
    X_signals.append(
        [np.array(serie, dtype=np.float32) for serie in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]]
    )
    file.close()

    return np.transpose(np.array(X_signals), (1, 2, 0))


def load_y(y_path):
    file = open(y_path, 'r')
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]],
        dtype=np.int32
    )
    file.close()

    return y_ - 1


def loadDataHMM():
    print('caricamento dei dati di training e test')
    X_train = produceMagnitude(0)
    X_test = produceMagnitude(1)

    y_train = load_y(y_train_path)
    y_test = load_y(y_test_path)

    # print('X_train', X_train)
    # print('y_train', y_train)
    # print('X_test', X_test)
    # print('y_test', y_test)

    print('fine caricamento')
    return X_train, y_train, X_test, y_test


def loadDataCNN():
    print('caricamento dei dati di training e test')
    X_train = load_X(X_train_signals_paths)
    X_test = load_X(X_test_signals_paths)

    y_train = load_y(y_train_path)
    y_test = load_y(y_test_path)

    # print('X_train', X_train)
    # print('y_train', y_train)
    # print('X_test', X_test)
    # print('y_test', y_test)

    print('fine caricamento')
    return X_train, y_train, X_test, y_test


def loadDataBLSTM():
    print('caricamento dei dati di training e test')
    X_train = load_X(X_train_signals_paths)
    X_test = load_X(X_test_signals_paths)

    y_train = load_y(y_train_path)
    y_test = load_y(y_test_path)

    # print('X_train', X_train)
    # print('y_train', y_train)
    # print('X_test', X_test)
    # print('y_test', y_test)

    print('fine caricamento')
    return X_train, y_train, X_test, y_test


def dataProcessingHMM(X_train, y_train, X_test, y_test):
    print('elaborazione dei dati...')
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    X_train = X_train.reshape((X_train.shape[0] * X_train.shape[1]), X_train.shape[2])
    # print(X_train.shape)

    X_test = X_test.reshape((X_test.shape[0] * X_test.shape[1]), X_test.shape[2])
    # print(X_train.shape)
    X_val = X_val.reshape((X_val.shape[0] * X_val.shape[1]), X_val.shape[2])
    X_train = X_train.reshape(1, -1)
    X_test = X_test.reshape(1, -1)
    X_val = X_val.reshape(1, -1)
    print('fine elaborazione dati')
    return X_train, y_train, X_test, y_test, X_val, y_val


def dataProcessingCNN(X_train, y_train, X_test, y_test):
    print('elaborazione dei dati...')
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    enc = enc.fit(y_train)

    y_train = enc.transform(y_train)
    y_test = enc.transform(y_test)
    y_val = enc.transform(y_val)

    print('dimensione reshape', X_val[..., np.newaxis].shape)

    X_train = X_train.reshape(6488, 561, 1, 1)
    X_test = X_test.reshape(3090, 561, 1, 1)
    X_val = X_val.reshape(721, 561, 1, 1)

    print('fine elaborazione dati')
    return X_train, y_train, X_test, y_test, X_val, y_val


def dataProcessingBLSTM(X_train, y_train, X_test, y_test):
    print('elaborazione dei dati...')

    X = np.concatenate((X_train, X_test))

    y = np.concatenate((y_train, y_test))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    enc = enc.fit(y_train)

    y_train = enc.transform(y_train)
    y_test = enc.transform(y_test)
    y_val = enc.transform(y_val)
    print('fine elaborazione dati')

    return X_train, y_train, X_test, y_test, X_val, y_val


def PlotHMM(length_train, length_val, length_test, train_scores, test_scores, val_scores):
    print('Inizio plotting Hidden Markov Model')
    plt.figure(figsize=(7, 5))
    plt.scatter(np.arange(length_train), train_scores, c='b', label='trainset')
    plt.scatter(np.arange(length_train, length_val), val_scores, c='r', label='testset - imitation')
    plt.scatter(np.arange(length_val, length_test), test_scores, c='g', label='testset - original')
    plt.title('Feature')
    plt.legend(loc='lower right')
    print('Fine plotting')

    plt.savefig(hmmGraph)
    plt.show()


def plotScatterHMM():
    print('Creazione dello Scatter Plot')
    sns.set()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    xs = np.arange(len(logdata))

    masks = hidden_states == 0
    ax.scatter(xs[masks], logdata[masks], c='red', label='WalkingDwnStairs')

    masks = hidden_states == 1
    ax.scatter(xs[masks], logdata[masks], c='blue', label='WalkingUpstairs')

    masks = hidden_states == 2
    ax.scatter(xs[masks], logdata[masks], c='green', label='Sitting')

    masks = hidden_states == 3
    ax.scatter(xs[masks], logdata[masks], c='yellow', label='Standing')

    masks = hidden_states == 4
    ax.scatter(xs[masks], logdata[masks], c='orange', label='Walking')

    masks = hidden_states == 5
    ax.scatter(xs[masks], logdata[masks], c='black', label='Jogging')

    # decommentare per congiungere tutti i punti sul grafico
    # ax.plot(xs, logdata, c='k')

    ax.set_xlabel('Indice')
    ax.set_ylabel('Valore sensore')
    fig.subplots_adjust(bottom=0.2)
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=True)
    fig.set_size_inches(800 / mydpi, 800 / mydpi)
    fig.savefig(filename1)
    fig.clf()
    print('ScatterPlot creato')


def plotDistributionHMM(self):
    print('Creazione grafico di distribuzione')
    # calcolo della distribuzione stazionaria
    eigenvals, eigenvecs = np.linalg.eig(np.transpose(P))
    one_eigval = np.argmin(np.abs(eigenvals - 1))
    pi = eigenvecs[:, one_eigval] / np.sum(eigenvecs[:, one_eigval])

    x_0 = np.linspace(mus[0] - 4 * sigmas[0], mus[0] + 4 * sigmas[0], 10000)
    fx_0 = pi[0] * ss.norm.pdf(x_0, mus[0], sigmas[0])

    x_1 = np.linspace(mus[1] - 4 * sigmas[1], mus[1] + 4 * sigmas[1], 10000)
    fx_1 = pi[1] * ss.norm.pdf(x_1, mus[1], sigmas[1])

    # x_2 = np.linspace(mus[2] - 4 * sigmas[2], mus[2] + 4 * sigmas[2], 10000)
    # fx_2 = pi[2] * ss.norm.pdf(x_2, mus[2], sigmas[2])

    # x_3= np.linspace(mus[3] - 4 * sigmas[3], mus[3] + 4 * sigmas[1], 10000)
    # fx_3 = pi[3] * ss.norm.pdf(x_3, mus[3], sigmas[3])

    # x_4= np.linspace(mus[4] - 4 * sigmas[4], mus[4] + 4 * sigmas[4], 10000)
    # fx_4 = pi[4] * ss.norm.pdf(x_4, mus[4], sigmas[4])

    # x_5 = np.linspace(mus[5] - 4 * sigmas[5], mus[5] + 4 * sigmas[1], 10000)
    # fx_5 = pi[5] * ss.norm.pdf(x_5, mus[5], sigmas[5])

    x = np.linspace(mus[0] - 4 * sigmas[0], mus[1] + 4 * sigmas[1], 10000)
    fx = pi[0] * ss.norm.pdf(x, mus[0], sigmas[0]) + pi[1] * ss.norm.pdf(x, mus[1], sigmas[1])

    sns.set()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(logdata, color='k', alpha=0.5, density=True)
    l1, = ax.plot(x_0, fx_0, c='red', linewidth=2, label='WalkingDwnStairs Distn')
    l2, = ax.plot(x_1, fx_1, c='blue', linewidth=2, label='WalkingUpStairs Distn')
    # l3, = ax.plot(x_2, fx_2, c='green', linewidth=2, label='Sitting Distn')
    # l4, = ax.plot(x_3, fx_3, c='yellow', linewidth=2, label='Standing Distn')
    # l5, = ax.plot(x_4, fx_4, c='orange', linewidth=2, label='Walking Distn')
    # l6, = ax.plot(x_5, fx_5, c='black', linewidth=2, label='Jogging Distn')
    l7, = ax.plot(x, fx, c='cyan', linewidth=2, label='Combined Distn')

    fig.subplots_adjust(bottom=0.15)
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=True)
    fig.set_size_inches(800 / mydpi, 800 / mydpi)
    fig.savefig(filename2)
    fig.clf()

    print('Fine creazione grafico')


def plot_learningCurveCNN(history, epochs):

    # Plot training & validation accuracy values
    plt.figure(figsize=(15, 8))
    epoch_range = range(1, epochs + 1)
    plt.plot(epoch_range, history.history['accuracy'])
    plt.plot(epoch_range, history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(trainingValAccCNN)
    # plt.show()

    # Plot training & validation auc values
    plt.figure(figsize=(15, 8))
    epoch_range = range(1, epochs + 1)
    plt.plot(epoch_range, history.history['auc'])
    plt.plot(epoch_range, history.history['val_auc'])
    plt.title('Model auc')
    plt.ylabel('auc')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(TrainingValAucCNN)
    # plt.show()

    # Plot training & validation loss values
    plt.figure(figsize=(15, 8))
    plt.plot(epoch_range, history.history['loss'])
    plt.plot(epoch_range, history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(ModelLossCNN)
    # plt.show()

def plot_learningCurveBLSTM(history, epochs):
    # Plot training & validation accuracy values
    plt.figure(figsize=(15, 8))
    epoch_range = range(1, epochs + 1)
    plt.plot(epoch_range, history.history['accuracy'])
    plt.plot(epoch_range, history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(trainingValAccBLSTM)
    # plt.show()

    # Plot training & validation auc values
    plt.figure(figsize=(15, 8))
    epoch_range = range(1, epochs + 1)
    plt.plot(epoch_range, history.history['auc'])
    plt.plot(epoch_range, history.history['val_auc'])
    plt.title('Model auc')
    plt.ylabel('auc')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(TrainingValAucBLSTM)
    # plt.show()

    # Plot training & validation loss values
    plt.figure(figsize=(15, 8))
    plt.plot(epoch_range, history.history['loss'])
    plt.plot(epoch_range, history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(ModelLossBLSTM)
    # plt.show()


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = loadDataHMM()
    X_train, y_train, X_test, y_test, X_val, y_val = dataProcessingHMM(X_train, y_train, X_test, y_test)

    print(X_train)
