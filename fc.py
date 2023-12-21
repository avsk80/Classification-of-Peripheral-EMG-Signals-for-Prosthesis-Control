from keras import models, layers, initializers
import tensorflow as tf
import keras
import data

default_input_shape_ = (28, data.num_frames_per_datapoint, 1)
nb_classes_ = 7

def create_fc(input_shape=default_input_shape_, nb_classes=nb_classes_):
    model = keras.Sequential()
    # model.add(layers.AveragePooling2D(pool_size=(1, 30), strides=(1, 15), input_shape=input_shape, padding='valid'))
    # model.add(layers.BatchNormalization(input_shape=input_shape))
    # model.add(layers.Activation('tanh'))
    # model.add(layers.Dropout(0.5))

    model.add(layers.Conv2D(64, kernel_size=(2, 2), strides=(1, 1),
                           activation='relu', padding="same", kernel_initializer=initializers.RandomNormal(stddev=1),
                           bias_initializer=initializers.Zeros()))
    model.add(layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

    model.add(layers.Flatten())

    # model.add(layers.Dense(126, activation='tanh'))

    model.add(layers.Dense(105, activation='tanh'))

    model.add(layers.Dense(84, activation='tanh'))

    model.add(layers.Dense(63, activation='tanh'))

    # model.add(layers.Dense(42, activation='tanh'))
    #
    # model.add(layers.Dense(21, activation='tanh'))

    model.add(layers.Dense(nb_classes, activation='softmax'))

    optimizer_ = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer_,
                 metrics=['accuracy'])
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