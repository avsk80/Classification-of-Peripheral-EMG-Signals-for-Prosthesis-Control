# Data preprocessing
import pickle
import gzip
import sys
import numpy as np
import h5py
import scipy.io
import pandas as pd

import matplotlib.pyplot as plt

import keras
from keras import backend
# from keras import datasets
# import keras

import keras.utils.np_utils as utils
# from tensorflow.keras import utils

num_frames_per_datapoint = 12
frames_bt_datapoints = 6

class DATA():
    def __init__(self):
        num_classes = 10

        f = gzip.open('Data/mnist.pkl.gz', 'rb')
        if sys.version_info < (3,):
            data = pickle.load(f)
        else:
            data = pickle.load(f, encoding='bytes')
        f.close()
        (x_train, y_train), (x_test, y_test) = data
        print("x_train ", np.shape(x_train))
        print("y_train ", np.shape(y_train))
        print(" x_test ", np.shape(x_test))
        print(" y_test ", np.shape(y_test))
        print(  "x_test single row ", x_test[0][14])
        print(" y test first 10 ", y_test[0:10])

        # (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
        img_rows, img_cols = x_train.shape[1:]

        if backend.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        y_train = utils.to_categorical(y_train, num_classes)
        y_test = utils.to_categorical(y_test, num_classes)

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

class DATA_DB6():
    def __init__(self, load=False, data_header="S1_D1", model_type="lenet", frames=15):

        num_classes = 7
        if load:
            (x_train, y_train) = (np.load('Data/db6_preprocessed/' + model_type + '_x_train_' + data_header + '.npy'),
                                  np.load('Data/db6_preprocessed/' + model_type + '_y_train_' + data_header + '.npy'))
            (x_test, y_test) = (np.load('Data/db6_preprocessed/' + model_type + '_x_test_' + data_header + '.npy'),
                                np.load('Data/db6_preprocessed/' + model_type + '_y_test_' + data_header + '.npy'))
            self.input_shape = np.shape(x_train[0])
            self.num_classes = num_classes
            self.x_train, self.y_train = x_train, y_train
            self.x_test, self.y_test = x_test, y_test
            return

        # print(name+'3')

        (x_train, y_train) = get_data_db6('Data/db6/' + data_header + '_T1.mat',
                                          model_type=model_type, frames=frames)
        (x_test, y_test) = get_data_db6('Data/db6/' + data_header + '_T2.mat',
                                          model_type=model_type, frames=frames)

        # Add dummy dimension
        # x_train = np.expand_dims(x_train, axis=3)
        # y_train = np.expand_dims(y_train, axis=1)

        # print("x_train ", np.shape(x_train))
        # print("y_train ", np.shape(y_train))
        # print(" x_test ", np.shape(x_test))
        # print(" y_test ", np.shape(y_test))
        # print("x_test single row ", x_test[0][14])
        # print(" y test first 10 ", y_test[0:10])

        # (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
        # img_rows, img_cols = x_train.shape[1:]
        #
        # if backend.image_data_format() == 'channels_first':
        #     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        #     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        #     input_shape = (1, img_rows, img_cols)
        # else:
        #     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        #     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        #     input_shape = (img_rows, img_cols, 1)

        # x_test = x_train
        # y_test = y_train

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        # print(" x_test ", np.shape(x_test))
        # print(" y_test ", np.shape(y_test))
        # print("x_test single row ", x_test[0][14])
        # print(" y test first 10 ", y_test[0:10])


        # x_train -= 1500
        # x_test -= 1500
        # x_train /= 3000
        # x_test /= 3000

        y_train = utils.to_categorical(y_train, num_classes)
        y_test = utils.to_categorical(y_test, num_classes)



        print("final shapes:")
        print(np.shape(x_train), np.shape(y_train))
        np.save('Data/db6_preprocessed/'+model_type+'_x_train_' + data_header + '.npy', x_train)
        np.save('Data/db6_preprocessed/'+model_type+'_y_train_' + data_header + '.npy', y_train)
        np.save('Data/db6_preprocessed/'+model_type+'_x_test_' + data_header + '.npy', x_test)
        np.save('Data/db6_preprocessed/'+model_type+'_y_test_' + data_header + '.npy', y_test)

        # (x, y) = (np.load('x_train.npy'), np.load('y_train.npy'))
        #
        # print("loaded shapes:")
        # print(np.shape(x), np.shape(y))

        self.input_shape = (16, num_frames_per_datapoint, 1)
        self.num_classes = num_classes
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test


class DATA_SUGURU():
    def __init__(self, load=False, data_header="Data/suguru", test_day = 16, model_type="lenet", frame=30, step=5,
                 test_only=False, save_to_day=-1):

        print(data_header)

        num_classes = 22
        load_train_only = False
        _frame = frame
        _step = step

        if save_to_day >= 0:
            path = data_header + '/day' + str(save_to_day) + '/'
        else:
            path = data_header + '/'

        if load:
            (x_train, y_train) = (np.load(path + model_type + '_x_train.npy'),
                                  np.load(path + model_type + '_y_train.npy'))
            (x_test, y_test) = (np.load(path + model_type + '_x_test.npy'),
                                np.load(path + model_type + '_y_test.npy'))

            self.input_shape = np.shape(x_train[0])
            self.num_classes = num_classes
            self.x_train, self.y_train = x_train, y_train
            self.x_test, self.y_test = x_test, y_test
            return

        if not test_only:
            if not load_train_only:

                # print(name+'3')
                result, result_y = [], []
                i = 1
                for i in range(1, 23):
                    WS = pd.read_excel(data_header + '/M' + str(i) + '.xlsx', engine='openpyxl')
                    # WS = pd.read_excel('Data/suguru_preprocessed/D1M' + str(i) + '_extracted.xlsx', engine='openpyxl')
                    WS_np = np.array(WS)
                    # print(np.shape(WS_np))

                    j = 0
                    while j + _frame < len(WS_np):
                        result.append(WS_np[j:j + _frame])
                        result_y.append([i-1])
                        j += _step
                # print(np.shape(result), np.shape(result_y))
                result, result_y = np.array(result), np.array(result_y)

                (x_train, y_train) = result, result_y

                # Add dummy dimension
                # if model_type != "rnn2":
                #     x_train = np.expand_dims(x_train, axis=3)
            else:
                (x_train, y_train) = (np.load('Data/suguru_preprocessed/' + model_type + '_x_train.npy'),
                                      np.load('Data/suguru_preprocessed/' + model_type + '_y_train.npy'))
        else:
            (x_train, y_train) = np.zeros(1), np.zeros(1)

        # test day:
        result, result_y = [], []
        for i in range(1, 23):
            filename = data_header + '/day' + str(test_day) + '/M' + str(i) + '.xlsx'
            print(filename)
            # print(" Compiling test day data from", filename)
            WS = pd.read_excel(filename, engine='openpyxl')
            # WS = pd.read_excel('Data/suguru_preprocessed/D1M' + str(i) + '_extracted.xlsx', engine='openpyxl')
            WS_np = np.array(WS)
            # print("WS_np shape:", np.shape(WS_np))
            # print(np.shape(WS_np))

            j = 0
            # while j + _frame < len(WS_np):
            #     result.append(WS_np[j:j + _frame])
            #     result_y.append([i - 1])
            #     j += _step

            # if too little data to make 1 sample of frame specified, copy and append data
            while j == 0:
                WS_np = np.append(WS_np, WS_np, axis=0)
                while j + _frame < len(WS_np):
                    result.append(WS_np[j:j + _frame])
                    result_y.append([i - 1])
                    j += _step

        # print("x_test, y_test shapes:", np.shape(result), np.shape(result_y))
        (x_test, y_test) = result, result_y
        if test_only:
            (x_train, y_train) = x_test, y_test

        # Add dummy dimension
        if model_type != "rnn2":
            if not test_only:
                x_train = np.expand_dims(x_train, axis=3)
            # print("x_test shape:", np.shape(x_test))
            # print(x_test)
            x_test = np.expand_dims(x_test, axis=3)
        # y_train = np.expand_dims(y_train, axis=1)


        x_train = np.array(x_train).astype('float32')
        x_test = np.array(x_test).astype('float32')

        print("  x train, test shapes: ", np.shape(x_train), np.shape(x_test))

        # print(" x_test ", np.shape(x_test))
        # print(" y_test ", np.shape(y_test))
        # print("x_test single row ", x_test[0][14])
        # print(" y test first 10 ", y_test[0:10])


        # x_train -= 1500
        # x_test -= 1500
        # x_train /= 3000
        # x_test /= 3000

        y_train = utils.to_categorical(y_train, num_classes)
        y_test = utils.to_categorical(y_test, num_classes)


        if model_type == "rnn2":
            x_train = np.transpose(x_train, (0, 2, 1))
            x_test = np.transpose(x_test, (0, 2, 1))


        # data_header = "Data/suguru/0"
        print("final shapes:")
        print(np.shape(x_train), np.shape(y_train))

        if not test_only:
            np.save(path + model_type + '_x_train.npy', x_train)
            np.save(path + model_type + '_y_train.npy', y_train)
        np.save(path + model_type + '_x_test.npy', x_test)
        np.save(path + model_type + '_y_test.npy', y_test)

        # (x, y) = (np.load('x_train.npy'), np.load('y_train.npy'))
        #
        # print("loaded shapes:")
        # print(np.shape(x), np.shape(y))

        self.num_classes = num_classes

        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.input_shape = np.shape(self.x_train[0])

def get_data_db6(filename, model_type="lenet", plot=False, frames=15):
    print("filename to be opened: "+filename)
    mat = scipy.io.loadmat(filename)
    data = mat["emg"]
    y = mat["restimulus"]
    print("np.shape(data), np.shape(y): ", np.shape(data), np.shape(y))
    # i = len(y)
    # print(y[i-1][0])
    l = [0 for i in range(15)]
    for i in range(len(y)):
        j = y[i][0]
        if j==11:
            y[i][0]=2
        elif j==10:
            y[i][0]=5
        elif j==9:
            y[i][0]=7
        # else:
        #     y[k][0] -= 2
        l[y[i][0]]+=1
    print(l)
    # print("============================================")
    # print(data[35965:35967])
    # print("y[35965:35967]:", y[35965:35967])

    lists = {"l0": np.zeros((1, 16)),
             "l1": np.zeros((l[1], 16)),
             "l2": np.zeros((l[2], 16)),
             "l3": np.zeros((l[3], 16)),
             "l4": np.zeros((l[4], 16)),
             "l5": np.zeros((l[5], 16)),
             "l6": np.zeros((l[6], 16)),
             "l7": np.zeros((l[7], 16)),}
    i = [0 for j in range(8)]
    i_lim = [len(lists["l"+str(j)]) for j in range(8)]



    # prt=True
    for j in range(len(y)):
        _y = y[j][0]
        if _y != 0 and i[_y] < i_lim[_y]:
            _x = data[j]
            # if prt:
            #     print("-------------------")
            #     print(_x)
            #     print("y["+str(j)+"][0]:", _y)
            #     prt=False
            lists["l"+str(_y)][i[_y]] = _x
            i[_y] += 1

    print("np.shape(lists[\"l1\"]): ", np.shape(lists["l1"]))

    # split with overlap
    for j in range(1, len(lists)):
        b = np.array(lists["l" + str(j)])
        b_list = b.tolist()
        B = [b[i: i + num_frames_per_datapoint] for i in range(0, len(b), frames_bt_datapoints)]
        B_np = np.array(B[:-(int(num_frames_per_datapoint/frames_bt_datapoints))])
        print("np.shape(np.array(B_np)):", np.shape(B_np))
        lists["l" + str(j)] = B_np
    # b = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]])
    # # b.flatten()
    # b_list = b.tolist()
    # num_frames_per_datapoint = 3
    # frames_bt_datapoints = 2
    # B = [b[i: i + num_frames_per_datapoint] for i in range(0, len(b), frames_bt_datapoints)]
    # print(np.array(B[:-1]))



    # print("l1 len, ", len(lists["l1"]))
    # for k in range(1, 8):
    #     print("l1["+str(k)+"] :", lists["l"+str(k)+""][0])

    # split lists
    # for j in range(1, 8):
    #     _l = lists["l"+str(j)]
    #     print("np.shape(_l):", np.shape(_l))
    #     end_ind = num_frames_per_datapoint * int(len(_l) / num_frames_per_datapoint)
    #     _l = _l[0:, :end_ind]
        # print("   np.shape(_l):", np.shape(_l))
        # print("end_ind=", end_ind)
        # _d = np.split(_l[0:, :end_ind], end_ind / num_frames_per_datapoint, axis=0)
        # print("Truncated l"+str(j)+" shape: ", np.shape(_d))
        # lists["l"+str(j)] = _d


    # print("l1 shape, ", np.shape(lists["l1"]))

    # overlapped_dp_lists = {"l0": np.zeros((1, 16)),
    #                        "l1": np.zeros(int((l[1] - num_frames_per_datapoint) / frames_bt_datapoints)
    #                                       * frames_bt_datapoints, 16),
    #                        "l2": np.zeros(int((l[2] - num_frames_per_datapoint) / frames_bt_datapoints)
    #                                       * frames_bt_datapoints, 16),
    #                        "l3": np.zeros(int((l[3] - num_frames_per_datapoint) / frames_bt_datapoints)
    #                                       * frames_bt_datapoints, 16),
    #                        "l4": np.zeros(int((l[4] - num_frames_per_datapoint) / frames_bt_datapoints)
    #                                       * frames_bt_datapoints, 16),
    #                        "l5": np.zeros(int((l[5] - num_frames_per_datapoint) / frames_bt_datapoints)
    #                                       * frames_bt_datapoints, 16),
    #                        "l6": np.zeros(int((l[6] - num_frames_per_datapoint) / frames_bt_datapoints)
    #                                       * frames_bt_datapoints, 16),
    #                        "l7": np.zeros(int((l[7] - num_frames_per_datapoint) / frames_bt_datapoints)
    #                                       * frames_bt_datapoints, 16)}
    #
    # for j in range(1, 8):
    #     _l = lists["l"+str(j)]
    #     _ol = overlapped_dp_lists["l"+str(j)]
    #
    #     l_i = 0
    #     while l_i + num_frames_per_datapoint < len(_l):
    #         for ol_i in range(l_i, l_i+num_frames_per_datapoint):
    #             _ol[]
    #
    #         l_i+=frames_bt_datapoints


    # concatenate arrays into final data
    new_data = np.concatenate((lists["l1"],
                               lists["l2"],
                               lists["l3"],
                               lists["l4"],
                               lists["l5"],
                               lists["l6"],
                               lists["l7"]))

    _y = []

    if model_type == "rnn":
        for j in range(1, 8):
            _y += [(j - 1) for k in range(len(lists["l" + str(j)]))]
            print("len(_y):", len(_y))
        new_data = np.transpose(new_data, (0, 1, 2))
        # new_data = np.transpose(new_data, (0, 2, 1))
        print("new data: ", new_data[0])
    else:
        new_data = np.expand_dims(new_data, axis=3)
        for j in range(1, 8):
            _y += [[j-1] for k in range(len(lists["l"+str(j)]))]
            print("len(_y):", len(_y))
        new_data = np.transpose(new_data, (0, 2, 1, 3))

    # print("sanity check:", _y[5000], _y[10000], _y[15000], _y[20000], _y[25000], _y[30000], _y[35000])
    new_y = np.array(_y)
    print("np.shape(new_y):", np.shape(new_y))
    print("new shapes: ", np.shape(new_data), np.shape(new_y))

    # i=0
    # for j in range(len(y)):
    #     k = y[j][0]
    #     if k!=0:
    #         new_data[i]=data[j]
    #         new_y[i]=y[j]
    #         i+=1

    # print(new_data[0])
    # new_data = np.transpose(new_data, (0, 2, 1, 3))
    # print(new_data[0])
    # print("transposed new shapes: ", np.shape(new_data), np.shape(new_y))

    # print("-----------------------------------")
    # print("new_data[0]: ", new_data[0])
    # print("new_y[0]: ", new_y[0])

    return new_data, new_y



def get_data(filename, plot=False):
    f = h5py.File(filename, 'r')
    data = np.array(f.get('Data_ADC0'))
    data = data[:, 5000:]
    (data_dim, num_datapts) = np.shape(data)
    end_ind = num_frames_per_datapoint * int(num_datapts / num_frames_per_datapoint)
    print(end_ind)
    print(np.shape(data))
    print(np.shape(data[0:, :end_ind]))
    print("min, max =", data.min(), ", ", data.max())

    (dim, num_frames) = np.shape(data)

    if plot:
        lim=num_frames
        # lim=1500
        x = np.arange(0, lim)
        plt.title(filename)
        plt.xlabel("X axis")
        plt.ylabel("Y axis")
        y = data[0][:lim]+150
        plt.plot(x, y, color="yellow")
        y = data[5][:lim]+100
        plt.plot(x, y, color="red")
        y = data[12][:lim]+50
        plt.plot(x, y, color="green")
        y = data[-1][:lim]
        plt.plot(x, y, color="blue")
        plt.show()

    data = np.split(data[0:, :end_ind], end_ind / num_frames_per_datapoint, axis=1)
    return data


class DATA_MAT():
    def __init__(self, load=True):

        num_classes = 5
        name = ['a', 'b', 'c', 'd', 'e']
        ind = ['1', '2', '3']
        if load:
            (x_train, y_train) = (np.load('x_train.npy'), np.load('y_train.npy'))
            (x_test, y_test) = (x_train, y_train)
            self.input_shape = (20, num_frames_per_datapoint, 1)
            self.num_classes = num_classes
            self.x_train, self.y_train = x_train, y_train
            self.x_test, self.y_test = x_test, y_test
            return

        # print(name+'3')

        x_train = get_data('Data/'+name[0]+ind[0]+'.mat')
        (x, y, z) = np.shape(x_train)
        y_train = np.zeros(x)

        for i in range(len(name)):
            for j in range(len(ind)):
                filename_ = 'Data/' + name[i] + ind[j] + '.mat'
                print(filename_, i)
                if ((i, j) != (0, 0)):
                    new_x = get_data('Data/'+name[i]+ind[j]+'.mat')
                    x_train = np.concatenate((x_train, new_x), axis=0)
                    (x, y, z) = np.shape(new_x)
                    new_y = np.zeros(x)
                    for k in range(len(new_y)):
                        new_y[k] = i
                    y_train = np.concatenate((y_train, new_y))


        # x_train = np.concatenate((x_train, x_train), axis=1)
        # x_train = x_train[:, :28]


        # f = h5py.File('Data/e3.mat', 'r')
        # data = f.get('Data_ADC0')
        # # data = np.array(data)
        # data = get_data('Data/b3.mat')
        # print("shape(data) =", np.shape(data))
        # print("max =", data.max())
        # print("min =", data.min())

        # Add dummy dimension
        x_train = np.expand_dims(x_train, axis=3)
        y_train = np.expand_dims(y_train, axis=1)

        print("x_train ", np.shape(x_train))
        print("y_train ", np.shape(y_train))
        # print(" x_test ", np.shape(x_test))
        # print(" y_test ", np.shape(y_test))
        # print("x_test single row ", x_test[0][14])
        # print(" y test first 10 ", y_test[0:10])

        # (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
        # img_rows, img_cols = x_train.shape[1:]
        #
        # if backend.image_data_format() == 'channels_first':
        #     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        #     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        #     input_shape = (1, img_rows, img_cols)
        # else:
        #     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        #     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        #     input_shape = (img_rows, img_cols, 1)

        x_test = x_train
        y_test = y_train

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        # print(" x_test ", np.shape(x_test))
        # print(" y_test ", np.shape(y_test))
        # print("x_test single row ", x_test[0][14])
        # print(" y test first 10 ", y_test[0:10])


        # x_train -= 1500
        # x_test -= 1500
        # x_train /= 3000
        # x_test /= 3000

        y_train = utils.to_categorical(y_train, num_classes)
        y_test = utils.to_categorical(y_test, num_classes)



        print("final shapes:")
        print(np.shape(x_train), np.shape(y_train))
        np.save('x_train.npy', x_train)
        np.save('y_train.npy', y_train)

        (x, y) = (np.load('x_train.npy'), np.load('y_train.npy'))

        print("loaded shapes:")
        print(np.shape(x), np.shape(y))

        self.input_shape = (20, num_frames_per_datapoint, 1)
        self.num_classes = num_classes
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test