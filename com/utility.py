"""
File contenente tutte le funzioni di utility, come lettura dei dataset, creazione di file e tutto il resto
"""
import pandas as pd

# directory contenente i dataset da utilizzare per l'addestramento degli agenti
dataset = '/home/w1l50n/PycharmProjects/HMMandDBSCANAnomalyRecognition/com/dataset/'


# funzione che apre i dataset e restituisce i riferimenti
def openDataset(file):
    directory = dataset + file
    df = pd.read_csv(directory, sep=',', header=None)
    df.columns = ["Date", "ACTION", "c", "etc."]
    print(df)


def convertNameFile(file):
    """
    cambia il nome del file da .txt a .csv
    :param file: string nome del file da cambiare
    :return:
    """

    print('inizio conversione nome file')
    file1 = file.split('.')
    csvFile = file1[0]
    csvFile = csvFile + '.csv'
    print('fine conversione file')
    return csvFile


def convertDataset(file):
    """
    Questa funzione serve a convertire i file .txt generati dal SDBSCAN in file .csv da pandas convertendoli nel formato corretto
    :param file: string, contiene il nome del file da convertire
    :return:
    """
    print('inizio conversione dataset')
    directory = dataset + file

    # apro il file .txt e leggo tutte le righe
    with open(directory, 'r') as dat:
        lines = dat.readlines()
        for i in range(0, len(lines)):
            print(lines)

    # inizio conversione del nome del file e creazione del nuovo file csv
    csvFile = convertNameFile(file)
    newDirectory = dataset + csvFile
    # fine conversione del nome del file e creazione del nuovo file csv


"""
    # inizio lettura del file .csv
    read_file = pd.read_csv(newDirectory)
    read_file.to_csv(newDirectory, index=None)
    print(read_file)
    print('conversione dataset effettuata con successo')
"""

# sezione in cui si testeranno tutte le funzioni create
if __name__ == '__main__':
    print('Inizio prova funzioni')

    file = 'sensor_c1FolG72RBKuDV2RLq1b-P.txt'
    openDataset(file)
    # convertDataset(file)
