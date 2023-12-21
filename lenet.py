from keras import models, layers, initializers
# from objectives import lda_loss
import tensorflow as tf
import keras
import data

default_input_shape_ = (28, data.num_frames_per_datapoint, 1)

def create_lenet(nb_classes, input_shape=default_input_shape_,
                 inner_model_name='', outer_model_name=''):
    if inner_model_name == '':
        inner_model = keras.Sequential()
        inner_model.add(layers.Conv2D(32, kernel_size=(3, 4), strides=(1, 1),
                                activation='relu', padding="same",
                                kernel_initializer=initializers.RandomNormal(stddev=1),
                                bias_initializer=initializers.Zeros()))
        # inner_model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))

        inner_model.add(layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                                activation='relu', padding='valid',
                                kernel_initializer=initializers.RandomNormal(stddev=1),
                                bias_initializer=initializers.Zeros()))
        inner_model.add(layers.Dropout(0.15))
        inner_model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))

        inner_model.add(layers.Conv2D(32, kernel_size=(2, 1), strides=(1, 1),
                                activation='relu', padding='valid',
                                kernel_initializer=initializers.RandomNormal(stddev=1),
                                bias_initializer=initializers.Zeros()))

        inner_model.add(layers.Conv2D(64, kernel_size=(1, 3), strides=(1, 2),
                                      activation='relu', padding='valid',
                                      kernel_initializer=initializers.RandomNormal(stddev=1),
                                      bias_initializer=initializers.Zeros()))
        inner_model.add(layers.Dropout(0.15))
        inner_model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))

        # inner_model.add(layers.Conv2D(64, kernel_size=(1, 2), strides=(1, 1),
        #                               activation='relu', padding='valid',
        #                               kernel_initializer=initializers.RandomNormal(stddev=1),
        #                               bias_initializer=initializers.Zeros()))
        #
        # inner_model.add(layers.Conv2D(128, kernel_size=(2, 2), strides=(1, 1),
        #                               activation='relu', padding='valid',
        #                               kernel_initializer=initializers.RandomNormal(stddev=1),
        #                               bias_initializer=initializers.Zeros()))
        # inner_model.add(layers.Dropout(0.15))

        # inner_model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

        # inner_model.add(layers.Conv2D(40, kernel_size=(2, 2), strides=(1, 1),
        #                         activation='relu', padding='valid',
        #                         kernel_initializer=initializers.RandomNormal(stddev=1),
        #                         bias_initializer=initializers.Zeros()))
        # inner_model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))


        # model.add(layers.Conv2D(120, kernel_size=(5, 5), strides=(1, 1),
        #                        activation='tanh', padding='valid'))

        inner_model.add(layers.Flatten())
        # inner_model.add(layers.Dense(500))
        inner_model.add(layers.Dense(200))
        inner_model.add(layers.Dropout(0.25))
        inner_model.add(layers.Dense(300))
        inner_model.add(layers.Dropout(0.25))
        inner_model.add(layers.Dense(80))
        inner_model.add(layers.Dropout(0.25))
        inner_model.add(layers.Dense(32))
    else:
        inner_model = models.load_model(inner_model_name)

    if outer_model_name == '':
        outer_model = keras.Sequential()
        # outer_model.add(layers.Dense(64))
        # outer_model.add(layers.Dense(32))
        outer_model.add(layers.Dense(nb_classes, activation='softmax'))
    else:
        outer_model = models.load_model(outer_model_name)

    model = keras.Sequential([
        keras.Input(shape=input_shape),
        inner_model,
        outer_model
    ])
    # model.add(layers.AveragePooling2D(pool_size=(1, 30), strides=(1, 15), input_shape=input_shape, padding='valid'))
    # model.add(layers.BatchNormalization(input_shape=input_shape))
    # model.add(layers.Activation('tanh'))
    # model.add(layers.Dropout(0.5))

    # model.add(layers.Conv2D(120, kernel_size=(5, 5), strides=(1, 1),
    #                        activation='tanh', padding='valid'))


    # model.add(layers.Dense(128, activation='tanh'))
    # model.add(layers.Dense(96, activation='tanh'))
    # # model.add(layers.Dense(64, activation='tanh'))
    # # model.add(layers.Dense(48, activation='tanh'))
    # model.add(layers.Dense(nb_classes, activation='softmax'))

    optimizer_ = tf.keras.optimizers.Adam(learning_rate=0.0001)

    # The margin and n_components (number of components) parameter used in the loss function
    # n_components should be at most class_size-1
    # margin = 1.0
    # n_components = 20
    # model.compile(loss=lda_loss(n_components, margin), optimizer=optimizer_)

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer_,
                  metrics=['accuracy'])
    return model, inner_model, outer_model
