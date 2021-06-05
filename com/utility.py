from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
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

checkPointPath = absPath_ + '/checkpoint'
graphAccuracy = absPath_ + '/immagine/CNN/'

labelDict = {'WALKING': 0, 'WALKING_UPSTAIRS': 1, 'WALKING_DOWNSTAIRS': 2,
             'SITTING': 3, 'STANDING': 4, 'LAYING': 5}


def produceMagnitude(flag):
    magnitude = []
    if flag:

        x = load_X(pathToSignalTrain + nameXtrain)
        y = load_X(pathToSignalTrain + nameYtrain)
        z = load_X(pathToSignalTrain + nameZtrain)

    else:
        x = load_X(pathToSignalTest + nameXtest)
        y = load_X(pathToSignalTest + nameYtest)
        z = load_X(pathToSignalTest + nameZtest)

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


def dataProcessingHMM(X_train, y_train, X_test, y_test):
    print('elaborazione dei dati...')
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)


    X_train= X_train.reshape((X_train.shape[0]*X_train.shape[1]), X_train.shape[2])
    #print(X_train.shape)

    X_test = X_test.reshape((X_test.shape[0] * X_test.shape[1]), X_test.shape[2])
    #print(X_train.shape)

    #X_train, y_train, X_test, y_test = np.log(X_train), np.log(y_train), np.log(X_test), np.log(y_test)
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

    # X_train = X_train.reshape(6488, 561, 1, 1)
    # X_test = X_test.reshape(3090, 561, 1, 1)
    # X_val = X_val.reshape(721, 561, 1, 1)

    X_train = X_train.reshape(6488, 187, 3, 1)
    X_test = X_test.reshape(3090, 187, 3, 1)
    X_val = X_val.reshape(721, 187, 3, 1)

    print('fine elaborazione dati')

    return X_train, y_train, X_test, y_test, X_val, y_val


def PlotHMM():
    plt.figure(figsize=(7, 5))
    plt.scatter(np.arange(length_train), train_scores, c='b', label='trainset')
    plt.scatter(np.arange(length_train, length_val), val_scores, c='r', label='testset - imitation')
    plt.scatter(np.arange(length_val, length_test), test_scores, c='g', label='testset - original')
    plt.title(f'User: {user} | HMM states: {n_components} | GMM components: {n_mix}')
    plt.legend(loc='lower right')

    username = 'user_' + str(user)
    figname = username + "_comp_" + str(n_components) + "_mix_" + str(n_mix) + '.png'
    plt.savefig("./hmm_plots/" + str(username) + "/" + str(figname))
    plt.show()


def plot_learningCurveCNN(history, epochs):
    # Plot training & validation accuracy values
    epoch_range = range(1, epochs + 1)
    plt.plot(epoch_range, history.history['accuracy'])
    plt.plot(epoch_range, history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    # Plot training & validation auc values
    epoch_range = range(1, epochs + 1)
    plt.plot(epoch_range, history.history['auc_1'])
    plt.plot(epoch_range, history.history['val_auc_1'])
    plt.title('Model auc')
    plt.ylabel('auc')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(epoch_range, history.history['loss'])
    plt.plot(epoch_range, history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(graphAccuracy + 'GraphAccuracyCNN.png')
    plt.show()


if __name__ == '__main__':
    magnitude = produceMagnitude(pathToSignalTrain)
