"""
File contenente tutte le funzioni di utility, come lettura dei dataset, creazione di file e tutto il resto
"""
import pandas as pd

# directory contenente i dataset da utilizzare per l'addestramento degli agenti
dataset = '/home/w1l50n/PycharmProjects/HMMandDBSCANAnomalyRecognition/com/dataset/'


# funzione che apre i dataset e restituisce i riferimenti
def openDataset(file):
    directory = dataset + file
    data = pd.read_csv(directory)  # inserisci indirizzo dataset
    print(data)


def convertDataset(file):
    print('inizio conversione dataset')

    directory = dataset + file
    newDirectory = dataset + 'newFile.csv'
    read_file = pd.read_csv(directory)
    read_file.to_csv(newDirectory, index=None)


# sezione in cui si testeranno tutte le funzioni create
if __name__ == '__main__':
    print('Inizio prova funzioni')

    file = 'sensor_fyXVCYrYSRyGivyIc9T3hN.txt'
    #openDataset(file)
    convertDataset(file)