"""
File contenente tutte le funzioni di utility, come lettura dei dataset, elaborazione degli stessi ed estrazione delle colonne contenenti i dati che utilizzeremo per allenare e testare gli HMM
"""

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# variabile globale che indica il dataset del tipo di attività che si vuole esaminare
testFilename_ = '/test.csv'
trainFilename_ = '/train.csv'

# serve a conoscere l'absolute path della cartella in cui si trova il file utility
absPath_ = os.getcwd()


# posizione delle cartelle dei vari dataset


class Dataset():

    def __init__(self):
        print('Testset caricato')
        print(absPath_)
        self.datasetPath = absPath_ + '/train_dataset/'

        self.labelDict = {'WALKING': 0, 'WALKING_UPSTAIRS': 1, 'WALKING_DOWNSTAIRS': 2, 'SITTING': 3, 'STANDING': 4,
                          'LAYING': 5}

        # carico i testset
        self.testset = pd.read_csv(self.datasetPath + testFilename_, index_col=0)

        # carico i trainset
        self.trainset = pd.read_csv(self.datasetPath + trainFilename_, index_col=0)

        # provo stampa dei testset e trainset
        self.stampaTestset()
        self.stampaTrainset()

    def stampaTestset(self):
        # viene effettuata la stampa del dataset caricato
        print('Stampa del testset')
        print(self.testset.head())

    def stampaTrainset(self):
        # viene effettuata la stampa del dataset caricato
        print('Stampa del trainset')
        print(self.trainset.head())

    def produceMagnitude(self, column):
        # controllare se serve ancora o posso utilizzarla in qualche maniera
        self.df[column + '.mag'] = np.sqrt(
            self.df[column + '.x'] ** 2 + self.df[column + '.y'] ** 2 + self.df[column + '.z'] ** 2)

    def loadTestset(self):
        self.stampaTestset()
        self.stampaTrainset()

        # sono tutti file in cui si caricheranno i dati di testset modificati
        train_X = self.trainset.values[:, :-2]

        train_y = self.trainset['Activity']
        train_y = train_y.map(self.labelDict).values

        test_X = self.testset.values[:, :-2]

        test_y = self.testset
        test_y = test_y['Activity'].map(self.labelDict).values

        return train_X, train_y, test_X, test_y

    def main(self):
        print('Sto caricando i file necessari...')

        # self.stampaTestset()
        # self.stampaTrainset()

        trainAndTest = self.loadTestset()
        print('Finito!')

        return trainAndTest


# sezione in cui si testeranno tutte le funzioni create


if __name__ == '__main__':
    ds = Dataset()
    trainAndTest = ds.main()

    for i in range(0, 3):
        print(trainAndTest[i])
