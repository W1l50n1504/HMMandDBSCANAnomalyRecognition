import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from scipy import stats as ss
from tensorflow.keras import *

from utility import *

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


def fitBLSTM():
    X_train, y_train, X_test, y_test = loadDataBLSTM()
    X_train, y_train, X_test, y_test, X_val, y_val = dataProcessingBLSTM(X_train, y_train, X_test, y_test)

    model = Sequential()
    model.add(layers.Bidirectional(
        layers.LSTM(units=64, return_sequences=True, input_shape=[X_train.shape[1], X_train.shape[2]])))

    model.add(layers.Dropout(rate=0.1))

    model.add(layers.Bidirectional(layers.LSTM(units=128)))

    model.add(layers.Dense(units=256, activation='relu'))

    model.add(layers.Dropout(rate=0.5))

    model.add(layers.Dense(y_train.shape[1], activation='softmax'))

    # model = load_model(checkPointPathBLSTM + '/best_model.hdf5',)

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC()])

    checkpoint = ModelCheckpoint(
        checkPointPathBLSTM + '/best_model.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto',
        period=1)

    history = model.fit(X_train, y_train, batch_size=16, epochs=10, validation_data=(X_test, y_test), verbose=1,
                        callbacks=[checkpoint])

    return history


if __name__ == '__main__':
    history = fitBLSTM()
    plot_learningCurveBLSTM(history, 10)
