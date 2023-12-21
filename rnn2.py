from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, SimpleRNN, Dropout, LSTM, GRU
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.datasets import mnist
from keras import models, layers, initializers

import tensorflow as tf
import keras
import data


dropout = 0.2

default_input_shape_ = (28, data.num_frames_per_datapoint, 1)
num_classes_ = 7

def create_rnn(input_shape=default_input_shape_, n_hidden=64, nb_classes=num_classes_,
               n_layers=4, inner_model_name=''):
    if inner_model_name == '':
        inner_model = keras.Sequential()
        rnn_params = n_hidden

        if n_layers <= 1:
            inner_model.add(LSTM(rnn_params, input_shape=input_shape))
            inner_model.add(Dropout(0.2))
        else:
            inner_model.add(LSTM(rnn_params, input_shape=input_shape, activation='relu', return_sequences=True))
            inner_model.add(Dropout(0.2))
            for i in range(n_layers - 2):
                rnn_params = int(0.8 * rnn_params)
                inner_model.add(tf.keras.layers.LSTM(units=rnn_params, return_sequences=True))
                inner_model.add(Dropout(0.2))
            inner_model.add(LSTM(n_hidden))
        inner_model.add(layers.Dense(200))
        inner_model.add(layers.Dropout(0.25))
        inner_model.add(layers.Dense(300))
        inner_model.add(layers.Dropout(0.25))
        inner_model.add(layers.Dense(80))
        inner_model.add(layers.Dropout(0.25))
        inner_model.add(layers.Dense(32))
    else:
        inner_model = models.load_model(inner_model_name)

    model = keras.Sequential([
        keras.Input(shape=input_shape),
        inner_model,
        # Dense(64, activation='tanh'),
        # Dense(32, activation='tanh'),
        Dense(nb_classes),
        Activation('softmax')
    ])
    # model.add(tf.keras.layers.GRU(32, input_shape=input_shape, name='gru'))
    # model.add(SimpleRNN(units=n_hidden,
    #                     dropout=dropout,
    #                     input_shape=input_shape))
    print("RNN input shape:", input_shape)

    # model.add(SimpleRNN(units=n_hidden,
    #                     dropout=dropout,
    #                     input_shape=input_shape))

    # model.add(GRU(units=n_hidden, dropout=dropout, input_shape=input_shape))


    # model.add(Dense(96, activation='tanh'))
    # model.add(Dense(64, activation='tanh'))
    # model.add(Dense(nb_classes))
    # model.add(Activation('softmax'))
    # model.summary()
    # enable this if pydot can be installed
    # pip install pydot
    # plot_model(model, to_file='rnn-mnist.png', show_shapes=True)

    # loss function for one-hot vector
    # use of sgd optimizer
    # accuracy is good metric for classification tasks
    optimizer_ = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer_,
                  metrics=['accuracy'])

    # model.compile(optimizer=optimizer_,
    #               metrics=['accuracy'])

    return model, inner_model