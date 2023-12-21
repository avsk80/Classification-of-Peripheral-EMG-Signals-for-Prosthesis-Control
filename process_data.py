from data import DATA
from data import DATA_MAT
from data import DATA_DB6
import data
import lenet
from plot import plot_loss
import matplotlib.pyplot as plt
from plot_keras_history import show_history, plot_history
import numpy as np
import keras
import tensorflow as tf
from keras.models import load_model, model_from_json

load = False
load_model_name = 'acc_65_val_54'


# input_shape_ = (28, 28, 1)
nb_classes_ = 10
input_shape_ = (20, data.num_frames_per_datapoint, 1)
nb_classes_ = 5

def main():
    # a = [j for j in range(6)]
    # b = np.reshape(np.array(a), (1, 2, 3))
    # print(b)
    # b=np.transpose(b, (0, 2, 1))
    # print(b)
    # print(np.shape(b))

    # process data
    subject_num = 1
    day = 1
    test_day = 1

    # Available model_types for preprocessing:
    # lenet, fc, rnn
    _model_type = "lenet"
    _data_header = _model_type + "_S" + str(subject_num) + "_D" + str(day)
    _test_data_header = _model_type + "_S" + str(subject_num) + "_D" + str(test_day)

    # Load single subject/day pair
    # data = DATA_DB6(load=False, model_type=_model_type,
    #                 data_header="S" + str(subject_num) + "_D" + str(day))


    # Load subjects 1-5 for days 1-5
    for subject_num in range(1, 2):
        if subject_num!=2:
            for day in range(1, 2):
            # for day in range(2, 6):
                data = DATA_DB6(data_header="S"+str(subject_num)+"_D"+str(day), model_type=_model_type)
    # subject_num = 2
    # for day in range(1, 6):
    #     if day != 2:
    #         data = DATA_DB6(data_header="S"+str(subject_num)+"_D"+str(day), model_type=_model_type)



    # data = DATA_DB6(load=True, model_type=_model_type,
    #                 data_header="S" + str(subject_num) + "_D" + str(day))
    # print(data.x_train[0])

    # print(data.x_train[0])
    # print("np.shape(data.x_train[0]): ", np.shape(data.x_train[0]))
    # for subject_num in range(3, 5):
    #     for day in range(1, 3):
    #         data = DATA_DB6(data_header="S"+str(subject_num)+"_D"+str(day))


if __name__ == '__main__':
    main()