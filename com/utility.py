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

walkDown = 'dws/'
walkUp = 'ups/'
sit = 'sit/'
stand = 'std/'
walk = 'wlk/'
jogging = 'jog/'

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
        #print('Carico il Dataset...')
        self.datasetPath = absPath_ + '/dataset/' + choice
        # carico il dataset
        self.ds = pd.read_csv(self.datasetPath + dataFilename_, index_col=0)
        self.dsm = pd.DataFrame()
        #print('Dataset caricato')

    def stampaDataset(self):
        # viene effettuata la stampa del dataset caricato
        print('Stampa del dataset')
        print(self.ds.head())

    def stampaTestset(self):
        # viene effettuata la stampa del testset
        print('Stampa del testset')
        print(self.dsm.head())

    def produceMagnitude(self):
        self.ds[magnitude] = np.sqrt(
            self.ds[column1 + '.x'] ** 2 + self.ds[column1 + '.y'] ** 2 + self.ds[
                column1 + '.z'] ** 2)
        #print('Magnitudine aggiunta correttamente nel dataset')

    def setTestset(self):
        self.dsm = self.ds[magnitude]

    def getTestset(self):
        return self.dsm

    def toList(self):
        return self.dsm.tolist()

    def main(self):
        # self.stampaDataset()
        self.produceMagnitude()  # calcola una nuova feature del dataset in cui attraverso una formula si ottiene un nuovo valore che permetta di tener conto delle variazioni che avvengono sui 3 assi x,y e z

        # self.stampaDataset()
        self.setTestset()  # copia la colonna creata della magnitudine in un nuovo dataset
        # self.stampaTestset()
        #print('Fine elaborazione dati')


if __name__ == '__main__':
    ds1 = Dataset(walkDown)
    ds1.main()
    lista = ds1.toList()
    print('stampo lista', lista)

    print('stampo testset senza funzione\n', ds1.getTestset())
