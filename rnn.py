from keras import models, layers, initializers
import tensorflow as tf
import keras
import data
from tensorflow.keras.layers import Dense, Dropout, LSTM

default_input_shape_ = (28, data.num_frames_per_datapoint, 1)
num_classes_ = 7

def create_rnn(input_shape, n_hidden=128, nb_classes=num_classes_, n_layers=5):
    model = keras.Sequential()
    # model.add(tf.keras.layers.GRU(32, input_shape=input_shape, name='gru'))
    model.add(LSTM(n_hidden, batch_input_shape=input_shape, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    for i in range(n_layers):
        model.add(tf.keras.layers.GRU(name='GRU_{}'.format(i + 1),
                                      units=n_hidden,
                                      activation='relu',
                                      stateful=True,
                                      return_sequences=(i < n_layers - 1)))
        model.add(Dropout(0.2))
    # model.add(tf.keras.layers.Dense(units=128, activation='softmax'))
    model.add(tf.keras.layers.Dense(units=nb_classes, activation='softmax'))  # class logits

    # track top 3 accuracy (= how often the true item is among the top 3 recommended)
    top3accuracy = lambda y_true, y_pred: tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)
    top3accuracy.__name__ = 'top3accuracy'
    model.compile(loss=keras.losses.binary_crossentropy, optimizer='adam', metrics=['accuracy', top3accuracy])

    model.build(input_shape=(data.num_frames_per_datapoint, 1, nb_classes))
    return model

#
# class LeNet(models.Sequential):
#     def __init__(self, input_shape=default_input_shape_, nb_classes=nb_classes_, name='name'):
#         super().__init__()
#
#         self.add(layers.AveragePooling2D(pool_size=(1, 30), strides=(1, 15), input_shape=input_shape, padding='valid'))
#         self.add(layers.Conv2D(32, kernel_size=(4, 4), strides=(1, 1),
#                                activation='tanh', padding="same", kernel_initializer=initializers.RandomNormal(stddev=300),
#     bias_initializer=initializers.Zeros()))
#         self.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
#         self.add(layers.Conv2D(16, kernel_size=(2, 2), strides=(1, 1),
#                                activation='tanh', padding='valid', kernel_initializer=initializers.RandomNormal(stddev=0.1),
#     bias_initializer=initializers.Zeros()))
#         self.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
#         # self.add(layers.Conv2D(120, kernel_size=(5, 5), strides=(1, 1),
#         #                        activation='tanh', padding='valid'))
#         self.add(layers.Flatten())
#         self.add(layers.Dense(84, activation='relu'))
#         self.add(layers.Dense(nb_classes, activation='softmax'))
#
#         # self.add(layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
#         #                        activation='tanh', input_shape=input_shape, padding="same"))
#         # self.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
#         # # self.add(layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1),
#         # #                        activation='tanh', padding='valid'))
#         # # self.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
#         # self.add(layers.Conv2D(120, kernel_size=(5, 5), strides=(1, 1),
#         #                        activation='tanh', padding='valid'))
#         # self.add(layers.Flatten())
#         # self.add(layers.Dense(84, activation='tanh'))
#         # self.add(layers.Dense(nb_classes, activation='softmax'))
#
#         n = 0
#         for layer in self.layers:
#             layer._name = "l"+str(n)
#             n+=1
#
#         optimizer_ = keras.optimizers.Adam(lr=0.001)
#
#         self.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer_,
#                      metrics=['accuracy'])