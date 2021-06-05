
class Dataset:

    def __init__(self, choice):
        """
        costruttore della classe Dataset
        :param choice: string, indica il percorso da utilizzare per caricare il dataset che si è scelto
        """
        print('Carico il Dataset...')
        self.datasetPath = absPath_ + choice
        # print(self.datasetPath)
        self.ds = pd.read_csv(self.datasetPath, index_col=0)
        print('Dataset caricato')

        self.dsm = pd.DataFrame()
        # print(self.ds)

    def stampaDataset(self):
        """
         viene effettuata la stampa del dataset caricato
        :return: None
        """

        print('Stampa del dataset\n')
        print(self.ds.head())

    def stampaTestset(self):
        """
         viene effettuata la stampa del dataset caricato che ha subito delle modifiche,
         come il calcolo della magnitudine
        :return: None
        """
        # viene effettuata la stampa del testset
        print('Stampa del testset\n')
        print(self.dsm.head())

    def produceMagnitude(self):
        """
        metodo utile per accorpare i dati registrati lungo i tre assi dell'accelerometro
        eleva al quadrato i dato dei singoli assi, li somma e infine effettua la radice quadrata
        inserendo il risultato in una nuova colonna chiamata userAcceleration.mag
        :return:
        """
        self.ds[magnitude] = np.sqrt(
            self.ds[column1 + '.x'] ** 2 + self.ds[column1 + '.y'] ** 2 + self.ds[
                column1 + '.z'] ** 2)
        # print('Magnitudine aggiunta correttamente nel dataset')

    def setTestset(self):
        """
        copia semplicemente i dati contenuti nella colonna userAcceleration.mag in un secondo dataset appartenente sempre alla classe
        :return:
        """
        self.dsm = self.ds[magnitude]

    def getTestset(self):
        """
        restituisce il dataset contenente solo la colonna userAcceleration.mag
        :return:
        """
        return self.dsm

    def toList(self):
        """
        serve per convertire il dataset caricato in memoria in una lista, verrà utilizzata per l'elaborazione dei dati (utilizzando la log-likelihood) durante
        il training del modello
        :return: list, lista di float
        """
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
    funzione utilizzata per ottenere il trainset.csv, semplicemente ho concatenato tutti i file sub_1.csv
    presenti in ogni cartella per creare un trainset su cui allenare il modello
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
