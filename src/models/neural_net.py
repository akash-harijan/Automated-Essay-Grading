import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization, regularizers, Dropout
import numpy as np


def create_model():

    model = Sequential()
    model.add(Dense(1024, input_shape=(300,), kernel_regularizer=regularizers.l2(0.0001)))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))

    model.add(Dense(512, kernel_regularizer=regularizers.l2(0.0001)))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))

    model.add(Dense(13, kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Activation('softmax'))
    model.summary()

    opt = keras.optimizers.sgd(lr=0.002)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy']
                  )
    return model
