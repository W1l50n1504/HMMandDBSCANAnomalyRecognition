"""
File contenente tutte le funzioni di utility, come lettura dei dataset, creazione di file e tutto il resto
spiegazione nomenclatura cartelle database:

    dws: walking downstairs
    ups: walking upstairs
    sit: sitting
    std: s  tanding
    wlk: walking
    jog: jogging

"""
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# rendi questa variabile globale
filename_ = 'dws_1/sub_1.csv'


class Dataset:

    def __init__(self):
        # serve a conoscere l'absolute path della cartella in cui si trova il file utility
        self.abs_path = os.getcwd()

        # posizione delle cartelle dei vari dataset

        self.combined_dataset = self.abs_path + '/train_dataset/A_DeviceMotion_data/'
        self.accelerometer_dataset = self.abs_path + '/train_dataset/B_Accelerometer_data/'
        self.gyscope_dataset = self.abs_path + '/train_dataset/C_Gyroscope_data/'

        self.df = 0

    def apriDataset(self, filename, dataset):
        # attraverso il numero passatogli  sceglie il dataset da caricare,
        # 0 quello contenente la combinazione dei dati sul giroscopio e l'accelerometro
        # 1 quello contenente i dati sull'accelerometro
        # 2 quello contenente i dati sul giroscopio
        if dataset == 0:
            self.df = pd.read_csv(os.path.join(self.combined_dataset, filename), index_col=0)

        elif dataset == 1:
            self.df = pd.read_csv(os.path.join(self.accelerometer_dataset, filename), index_col=0)

        elif dataset == 2:
            self.df = pd.read_csv(os.path.join(self.gyscope_dataset, filename), index_col=0)

        else:
            print("Non è stata inserita un'opzione valida")

    def stampaDataset(self):
        # viene effettuata la stampa del dataset caricato
        print(self.df.head())

    def produce_magnitude(self, column):
        # crea la nuova colonna contenente il vettore risultante dei tre registrati e presenti nel sistema
        self.df[column + '.mag'] = np.sqrt(
            self.df[column + '.x'] ** 2 + self.df[column + '.y'] ** 2 + self.df[column + '.z'] ** 2)


# sezione in cui si testeranno tutte le funzioni create
if __name__ == '__main__':
    # analisi del dataset  contenente i dati combinati
    print('Stampo i dati combinati')
    combined = Dataset()
    combined.apriDataset(filename_, 0)
    combined.stampaDataset()

    combined.produce_magnitude('userAcceleration')
    combined.produce_magnitude('rotationRate')
    combined.stampaDataset()

    # analisi del dataset  contenente i dati dell'accelerometro
    print('Stampo i dati accelerometro')
    accelerometer = Dataset()
    accelerometer.apriDataset(filename_, 1)
    accelerometer.stampaDataset()

    # analisi del dataset  contenente i dati del giroscopio
    print('Stampo i dati giroscopio')
    gyroscope = Dataset()
    gyroscope.apriDataset(filename_, 2)
    gyroscope.stampaDataset()
