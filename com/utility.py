import numpy as np
import pandas as pd
from keras import *
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils import to_categorical

trainset = '/dataset/train.csv'
testset = '/dataset/test.csv'

absPath_ = os.getcwd()
pathTrain = absPath_ + trainset
pathTest = absPath_ + testset

labelDict = {'WALKING': 0, 'WALKING_UPSTAIRS': 1, 'WALKING_DOWNSTAIRS': 2,
             'SITTING': 3, 'STANDING': 4, 'LAYING': 5}


def read_data(path):
    return pd.read_csv(path)


def load_dataset(labelDict_):
    train_X = read_data(pathTrain).values[:, :-2]
    train_y = read_data(pathTrain)['Activity']
    train_y = train_y.map(labelDict_).values

    test_X = read_data(pathTest).values[:, :-2]
    test_y = read_data(pathTest)
    test_y = test_y['Activity'].map(labelDict_).values

    return (train_X, train_y, test_X, test_y)


def produceMagnitude():
    """
    metodo utile per accorpare i dati registrati lungo i tre assi dell'accelerometro
    eleva al quadrato i dato dei singoli assi, li somma e infine effettua la radice quadrata
    inserendo il risultato in una nuova colonna chiamata userAcceleration.mag
    :return:
    """
    ds[magnitude] = np.sqrt(
        ds[column1 + '.x'] ** 2 + ds[column1 + '.y'] ** 2 + ds[
            column1 + '.z'] ** 2)
    # print('Magnitudine aggiunta correttamente nel dataset')


def encode(train_X, train_y, test_X, test_y):
    train_y = train_y - 1
    test_y = test_y - 1
    # one hot encode y
    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)

    print(train_X, train_y, test_X, test_y)

    return train_X, train_y, test_X, test_y


if __name__ == '__main__':
    df = read_data(pathTrain)
    # print(df.head())
    train_X, train_y, test_X, test_y = load_dataset(labelDict)
    #train_X, train_y, test_X, test_y = encode(train_X, train_y, test_X, test_y)

    print(train_X, train_y, test_X, test_y)
