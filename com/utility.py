"""
File contenente tutte le funzioni di utility, come lettura dei dataset, elaborazione degli stessi ed estrazione delle colonne contenenti i dati che utilizzeremo per allenare e testare gli HMM
il nome delle cartelle indica un'attività specifica:

    dws: walking downstairs
    ups: walking upstairs
    sit: sitting
    std: standing
    wlk: walking
    jog: jogging
"""

import os
import numpy as np
import pandas as pd

walkDown = '/dataset/dws/sub_1.csv'
walkUp = '/dataset/ups/sub_1.csv'
sit = '/dataset/sit/sub_1.csv'
stand = '/dataset/std/sub_1.csv'
walk = '/dataset/wlk/sub_1.csv'
jogging = '/dataset/jog/sub_1.csv'

trainset = '/dataset/trainset.csv'
testset = '/dataset/testset.csv'

column1 = 'userAcceleration'
# column2 = 'rotationRate'
magnitude = 'userAcceleration.mag'

# variabile globale che indica il dataset del tipo di attività che si vuole esaminare
dataFilename_ = 'sub_1.csv'

# serve a conoscere l'absolute path della cartella in cui si trova il file utility
absPath_ = os.getcwd()


# posizione delle cartelle dei vari dataset


class Dataset:

    def __init__(self, choice):
        print('Carico il Dataset...')
        self.datasetPath = absPath_ + choice
        # print(self.datasetPath)
        self.ds = pd.read_csv(self.datasetPath, index_col=0)
        print('Dataset caricato')

        self.dsm = pd.DataFrame()
        # print(self.ds)

    def stampaDataset(self):
        # viene effettuata la stampa del dataset caricato
        print('Stampa del dataset\n')
        print(self.ds.head())

    def stampaTestset(self):
        # viene effettuata la stampa del testset
        print('Stampa del testset\n')
        print(self.dsm.head())

    def produceMagnitude(self):
        self.ds[magnitude] = np.sqrt(
            self.ds[column1 + '.x'] ** 2 + self.ds[column1 + '.y'] ** 2 + self.ds[
                column1 + '.z'] ** 2)
        # print('Magnitudine aggiunta correttamente nel dataset')

    def setTestset(self):
        self.dsm = self.ds[magnitude]

    def getTestset(self):
        return self.dsm

    def toList(self):
        # return self.dsm.tolist()
        return self.ds.values.tolist()

    def main(self):
        # self.stampaDataset()
        self.produceMagnitude()
        # self.stampaDataset()
        self.setTestset()  # copia la colonna creata della magnitudine in un nuovo dataset
    # self.stampaTestset()
    # print('Fine elaborazione dati')


def unisciDiversiDataset():
    """
    funzione utilizzata per ottenere il trainset.csv, semplicemente ho unito tutti i file sub_1.csv presenti in ogni cartella per creare un trainset su cui allenare la ia
    """

    ds0 = Dataset(walkDown)
    ds1 = Dataset(walkUp)
    ds2 = Dataset(sit)
    ds3 = Dataset(stand)
    ds4 = Dataset(walk)
    ds5 = Dataset(jogging)

    ds0.main()
    ds1.main()
    ds2.main()
    ds3.main()
    ds4.main()
    ds5.main()

    # listds = [ds0.getTestset(), ds1.getTestset(), ds2.getTestset(), ds3.getTestset(), ds4.getTestset(),
    # ds5.getTestset()]
    # indice = ds0.dsm.shape[0] + ds1.dsm.shape[0] + ds2.dsm.shape[0] + ds3.dsm.shape[0] + ds4.dsm.shape[0] + \
    # ds5.dsm.shape[0]

    # index = np.arange(indice)

    listds = [ds0.dsm, ds1.dsm, ds2.dsm, ds3.dsm, ds4.dsm,
              ds5.dsm]

    # print('index\n',index)

    # pd.read_csv( absPath_ + '/dataset/' + testset, index_col=0)
    ds6 = pd.concat(listds).reset_index(drop=True)

    print('\nstampo ds6\n', ds6.head())

    ds6.to_csv(absPath_ + trainset, index=True)
    # print(ds6.head)


if __name__ == '__main__':
    #unisciDiversiDataset()
    ds = Dataset(trainset)

    lista = ds.toList()
    print('stampo dataset senza funzione\n', lista) #vuota per ora
