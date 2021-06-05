import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

from utility import *


def cnn(X_train, y_train, X_test, y_test, X_val, y_val, epochs):
    print('Inizio creazione CNN')

    model = Sequential()
    model.add(Conv2D(64, 1, activation='relu', input_shape=X_train[0].shape))
    model.add(Dropout(0.1))

    model.add(Conv2D(128, 1, activation='relu', padding='valid'))
    model.add(MaxPool2D(1, 1))

    model.add(Dropout(0.5))
    model.add(Flatten())

    model.add(Dense(512, activation='relu'))

    model.add(Dense(6, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['accuracy', tf.keras.metrics.AUC()])

    checkpoint = ModelCheckpoint(checkPointPath + '/best_model.hdf5',
                                 monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto', period=1)

    history = model.fit(X_train, y_train, batch_size=64, epochs=epochs, validation_data=(X_val, y_val),
                        verbose=1, callbacks=[checkpoint])
    return history


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = loadDataCNN()
    X_train, y_train, X_test, y_test, X_val, y_val, = dataProcessingCNN(X_train, y_train, X_test, y_test)
    history = cnn(X_train, y_train, X_test, y_test, X_val, y_val, 10)
    plot_learningCurve(history, 10)
