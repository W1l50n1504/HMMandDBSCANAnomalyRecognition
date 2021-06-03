import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from utility import *

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


class CNN:
    def __init__(self, train_X, train_y, test_X, test_y):
        # caricamento dei testset e trainset
        self.train = train_X.astype('float32')
        self.train_labels = train_y
        self.test = test_X.astype('float32')
        self.test_labels = test_y

        self.train_labels = labelDict

        self.test_labels = None
        self.history = None

        # crezione labels
        self.train_labels = []
        self.test_labels = ['WalkingDwnStairs', 'WalkingUpstairs', 'Sitting', 'Standing', 'Walking', 'Jogging']

        # creazione base convoluzionale
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(64, 1, activation='relu', input_shape=self.train[0].shape))
        self.model.add(layers.MaxPooling2D((1, 1)))
        self.model.add(layers.Conv2D(128, 1, activation='relu'))
        self.model.add(layers.MaxPooling2D((1, 1)))
        self.model.add(layers.Conv2D(128, 1, activation='relu'))

        # stampa architettura del modello
        self.model.summary()

        # spostamento degli stati densi in cima
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(10))

        # stampa architettura del modello
        self.model.summary()

    def fitting(self):
        # compilazione e addestramento del modello
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

        self.history = self.model.fit(self.train, self.train_labels, epochs=10,
                                      validation_data=(self.test, self.test_labels))

    def plot(self):
        # valutazione del modello
        plt.plot(self.history.history['accuracy'], label='accuracy')
        plt.plot(self.history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')

        test_loss, test_acc = self.model.evaluate(self.test, self.test_labels, verbose=2)
        print(test_acc)


if __name__ == '__main__':
    train_X, train_y, test_X, test_y = load_dataset(labelDict)

    # train_X, train_y, test_X, test_y = encode(train_X, train_y, test_X, test_y)
    # creazione del modello avvenuta con successo
    cnn = CNN(train_X, train_y, test_X, test_y)
    # problema nel fitting del modello, non accetta float in input
    cnn.fitting()
    # cnn.plot()
