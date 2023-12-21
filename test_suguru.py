from data import DATA
from data import DATA_MAT
from data import DATA_DB6
from data import DATA_SUGURU
import data
import lenet
import rnn
import fc
from plot import plot_loss
import matplotlib.pyplot as plt
from plot_keras_history import show_history, plot_history
import numpy as np
import keras
import tensorflow as tf
from keras.models import load_model, model_from_json
from sklearn.metrics import confusion_matrix
import time


load_data = False
load_model_name = 'lenet_S1_D1'

# input_shape_ = (28, 28, 1)
# nb_classes_ = 7
# input_shape_ = (16, data.num_frames_per_datapoint, 1)

input_shape_ = (252, data.num_frames_per_datapoint, 1)
nb_classes_ = 22


# process data
subject_num = 1
day = 1
test_day = 5

# Available models: lenet, fc, rnn
_model_type = "lenet"
_data_header = "S"+str(subject_num)+"_D"+str(day)
_test_data_header = "S"+str(subject_num)+"_D"+str(test_day)

def test(test_day, path="Data/suguru", load_model_name = load_model_name, model_type = _model_type):
    batch_size = 200
    epoch = 15

    # # process data
    # subject_num = 1
    # day = 1
    # test_day = 1
    #
    # # Available models: lenet, fc, rnn
    # _model_type = "fc"
    # _data_header = "S"+str(subject_num)+"_D"+str(day)
    # _test_data_header = "S"+str(subject_num)+"_D"+str(test_day)
    # for day in range(3, 6):
    #     data = DATA_DB6(data_header="S"+str(subject_num)+"_D"+str(day))
    # subject_num = 5
    # for day in range(1, 6):
    #     data = DATA_DB6(data_header="S" + str(subject_num) + "_D" + str(day))

    print("Testing: Loading model from " + load_model_name)

    model = load_model(load_model_name)


    # unique, counts = np.unique(y_train, return_counts=True)
    # print("unique, counts = ", unique, counts)

    print("Starting test, path =", path)

    # data_test = DATA_DB6(load=True, data_header=_test_data_header, model_type=_model_type)
    # data = DATA_SUGURU(load=load_train_data, data_header=data_header, model_type=_model_type,
    #                    frame=30, step=5)
    data_test = DATA_SUGURU(load=load_data, data_header=path, model_type=model_type,
                       frame=30, step=5, test_day=test_day, test_only=True)
    print(np.shape(data_test.x_train), np.shape(data_test.x_test))
    score = model.evaluate(data_test.x_test, data_test.y_test, batch_size=batch_size)
    # hist = model.fit(data_test.x_train, data_test.y_train, batch_size=batch_size, epochs=1,
    #                  validation_data=(data_test.x_test, data_test.y_test))
    # print("hist =", hist)
    # score = hist.history['accuracy'][-1]

    y_prediction = model.predict(data_test.x_test)

    print()
    print('Test Loss= ', score[0])
    print('Test Accuracy= ', score[1])

    # print(y_prediction[0])
    # print(data_test.y_test[0])

    y_pred_ = np.array([np.argmax(y_prediction[i]) for i in range(len(y_prediction))])
    y_test_ = np.array([np.argmax(data_test.y_test[i]) for i in range(len(data_test.y_test))])
    print("y_pred, y_test shapes: ", np.shape(y_pred_), np.shape(y_test_))
    result = confusion_matrix(y_test_, y_pred_, normalize='pred')
    # result = confusion_matrix(np.array([1, 2, 3, 4]), np.array([1, 2, 3, 1]), normalize='pred')
    # print('Confusion matrix:\n', result)
    np.savetxt("confusion_matrix.csv", result, delimiter=",")


    # plot_loss(hist)
    # plt.show()

    # history_dict = hist.history
    # train_acc = history_dict['loss']
    # val_acc = history_dict['val_loss']
    # epochs = range(1, len(history_dict['loss']) + 1)
    # plt.plot(epochs, train_acc, 'bo', label='Training Accuracy')
    # plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
    # plt.title('Training and Validation Accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.show()
    return score[1]

def main():
    t1 = time.perf_counter()
    result = []
    for i in range(1, 31):
        tt1 = time.perf_counter()
        print("\n\nTest day %d:" %i)
        result.append([i, test(i)])
        tt2 = time.perf_counter()
        print("Day %d test duration: %d sec" %(i, (tt2-tt1)))
    result.reverse()
    result = np.array(result)
    print("\n\n Result: \n", result)
    np.savetxt("result.csv", result.T, delimiter=",")
    t2 = time.perf_counter()
    print(" Time elapsed (min): ", int((t2-t1)/60))

if __name__ == '__main__':
    main()