import keras.layers
from keras.layers import Dense
from keras.models import Sequential
from keras.regularizers import l2
from keras import models, layers, initializers


def create_model(input_dim, reg_par, outdim_size):
    """
    Builds the model
    The structure of the model can get easily substituted with a more efficient and powerful network like CNN
    """
    model = Sequential()
    model.add(keras.layers.Conv2D(64, kernel_size=(2, 2), strides=(1, 1),
                            activation='relu', padding="same", kernel_initializer=initializers.RandomNormal(stddev=1),
                            bias_initializer=initializers.Zeros()))
    model.add(layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
    model.add(Dense(1024, input_shape=(input_dim,), activation='sigmoid', kernel_regularizer=l2(reg_par)))
    model.add(Dense(1024, activation='sigmoid', kernel_regularizer=l2(reg_par)))
    model.add(Dense(1024, activation='sigmoid', kernel_regularizer=l2(reg_par)))
    model.add(Dense(outdim_size, activation='linear', kernel_regularizer=l2(reg_par)))

    return model