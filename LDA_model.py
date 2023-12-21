from data import DATA
from data import DATA_MAT
from data import DATA_DB6
from data import DATA_SUGURU
import data
import lenet
import rnn
import rnn2
import fc
import time
from plot import plot_loss
import matplotlib.pyplot as plt
from plot_keras_history import show_history, plot_history
import numpy as np

# Force use CPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""


import keras
import tensorflow as tf
from keras.models import load_model, model_from_json
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from test_suguru import test
from suguru_combine_t import combine_t, compile_training

# outputs of models [m1,m2,m3,m4] -> classification (index)
classification_key = np.array([
    [0,0,0,-1],      # y=0
    [1,0,0,-1],      # y=1
    [-1,0,0,-1],      # y=2
    [0,1,0,-1],
    [0,-1,0,-1],
    [0,0,1,-1],      # y=5
    [0,0,-1,-1],
    [0,0,0,1],
    [1,0,1,-1],
    [-1,0,1,-1],
    [0,1,1,-1],      # y=10
    [0,-1,1,-1],
    [1,0,-1,-1],
    [-1,0,-1,-1],
    [0,1,-1,-1],
    [0,-1,-1,-1],      # y=15
    [1,0,0,1],
    [-1,0,0,1],
    [0,1,0,1],
    [0,-1,0,1],
    [0,0,1,1],      # y=20
    [0,0,-1,1],
])

# model1
# p = [x for x in range(1, 22)]
# n = [0]
#
# model2
# p = [1,8,12,16]
# n = [2,9,13,17]
#
# model3
# p = [3,10,14,18]
# n = [4,11,15,19]
#
# model4
# p = [5,8,9,10,11,20]
# n = [6,12,13,14,15,21]
#
# model5
# p = [7,16,17,18,19,20,21]
# n = []


class suguru_LDA_unit():
    def __init__(self, class_p1=[x for x in range(1, 22)], class_n1=[0], solver='svd'):
        # total_classes = np.array([x for x in range(22)])
        # class_p1 = np.array(class_p1)
        # class_n1 = np.array(class_n1)
        # existing_classes = np.append(class_n1, class_p1)
        # class_other = np.setdiff1d(total_classes, existing_classes)
        # self.classes = (class_p1, class_other, class_n1)

        self.model = LinearDiscriminantAnalysis(solver=solver)
        self.class_p1 = np.array(class_p1)
        self.class_n1 = np.array(class_n1)

    def _map_y(self, y):
        result_y = np.ndarray.copy(y)

        for i in range(len(result_y)):
            raw_val = result_y[i]
            if raw_val in self.class_p1:
                result_y[i]=1
            elif raw_val in self.class_n1:
                result_y[i]=-1
            else:
                result_y[i]=0

        return result_y

    def fit(self, X, y):
        result = self.model.fit(X, self._map_y(y))
        return result

    def predict(self, X):
        result = self.model.predict(X)
        return result

    def score(self, X, y):
        result = self.model.score(X, self._map_y(y))
        return result

class suguru_LDA_unit_mask():
    def __init__(self, class_list=[0], ignore=True, solver='svd'):
        # total_classes = np.array([x for x in range(22)])
        # class_p1 = np.array(class_p1)
        # class_n1 = np.array(class_n1)
        # existing_classes = np.append(class_n1, class_p1)
        # class_other = np.setdiff1d(total_classes, existing_classes)
        # self.classes = (class_p1, class_other, class_n1)
        if ignore:
            class_ignore = class_list
        else:
            class_ignore = np.setdiff1d(np.array([x for x in range(22)]),
                                        class_list)

        self.model = LinearDiscriminantAnalysis(solver=solver)
        self.class_ignore = np.array(class_ignore)

    def _map_input(self, X, y):
        delete_index_list = []

        for i in range(len(y)):
            if y[i] in self.class_ignore:
                delete_index_list.append(i)

        result_x = np.delete(np.ndarray.copy(X), delete_index_list, axis=0)
        result_y = np.delete(np.ndarray.copy(y), delete_index_list)

        return result_x, result_y

    def fit(self, X, y):
        new_x, new_y = self._map_input(X, y)
        result = self.model.fit(new_x, new_y)
        return result

    def predict(self, X):
        result = self.model.predict(X)
        return result

    def score(self, X, y):
        new_x, new_y = self._map_input(X, y)
        result = self.model.score(new_x, new_y)
        return result

class suguru_LDA():
    def __init__(self, solver='svd'):
        self.models = (suguru_LDA_unit(class_p1=[x for x in range(1, 22)], class_n1=[0], solver=solver),
                       suguru_LDA_unit(class_p1=[1,8,12,16], class_n1=[2,9,13,17], solver=solver),
                       suguru_LDA_unit(class_p1=[3,10,14,18], class_n1=[4,11,15,19], solver=solver),
                       suguru_LDA_unit(class_p1=[5,8,9,10,11,20], class_n1=[6,12,13,14,15,21], solver=solver),
                       suguru_LDA_unit(class_p1=[7,16,17,18,19,20,21],
                                       class_n1=[0,1,2,3,4,5,6,8,9,10,11,12,13,14,15], solver=solver))


    def predict(self, X):
        result = np.zeros(len(X))
        for i in range(len(X)):
            x = [X[i]]
            if self.models[0].predict(x)[0] < 0:
                result[i] = 0
            else:
                pred_v = np.array([m.predict(x)[0] for m in self.models[1:]])
                dot_res = [np.dot(v, pred_v) for v in classification_key]
                result[i] = np.argmax(dot_res)
        return result

    def _inner_linearity(self, X, y):
        result = np.array([m.score(X, y)
                           for m in self.models])
        return result

    def fit(self, X, y):
        for m in self.models:
            m.fit(X, y)
        return

    def score(self, X, y):
        y_pred = self.predict(X)
        result = accuracy_score(y, y_pred)
        return result



def revert_one_hot(y_oh):
    y_oh = np.array(y_oh)
    result = []
    for sample in y_oh:
        result.append(sample.argmax(axis=0))
    return np.array(result)


if __name__ == '__main__':
    path = "Data/suguru/37"
    data_1 = DATA_SUGURU(load=False, data_header=path, model_type="lenet",
                                   frame=30, step=5, test_day=16-15, save_to_day=16-15, test_only=True)
    (x1, y1) = (data_1.x_test, revert_one_hot(data_1.y_test))
    x1_flat = np.array([np.array(s).flatten() for s in x1])

    data_2 = DATA_SUGURU(load=False, data_header=path, model_type="lenet",
                       frame=30, step=5, test_day=16, save_to_day=16, test_only=True)
    (x2, y2) = (data_2.x_test, revert_one_hot(data_2.y_test))
    x2_flat = np.array([np.array(s).flatten() for s in x2])

    print("x1 flatten shape:", np.shape(x1_flat))
    x = np.append(x1, x2, axis=0)
    x_flat = np.array([np.array(s).flatten() for s in x])
    y = np.append(y1, y2)
    print("x, y shapes:", np.shape(x), np.shape(y))


    # Raw total linearity
    sug = suguru_LDA()
    # print(y)
    # print("mapped y\n", sug._map_y(y))
    sug.fit(x_flat, y)
    # print("new y\n", y)
    total_linearity = sug.score(x_flat, y)


    print(sug.predict(x_flat))
    print(y)
    print("total linearity =", total_linearity)
    print("inner linearity =\n", sug._inner_linearity(x_flat, y))

    clf = LinearDiscriminantAnalysis(solver='svd')
    clf.fit(x_flat, y)
    total_linearity = clf.score(x_flat, y)


    # print(clf.predict(x_flat))
    print("total linearity =", total_linearity)
    # print(y)





    b2 = np.array([1, 8, 12, 16])
    b3 = np.array([2, 9, 13, 17])
    b_diff = np.append(b2, b3)
    b4 = np.setdiff1d(np.array([x for x in range(1, 22)]), b_diff)

    print('\n\n\n\n')
    m = suguru_LDA_unit_mask(class_list=b2, ignore=False)
    x_new, y_new = m._map_input(x_flat, y)
    print("np.shape(x_flat), np.shape(y):", np.shape(x_flat), np.shape(y))
    print("np.shape(x_new), np.shape(y_new):", np.shape(x_new), np.shape(y_new))

    m.fit(x_flat, y)
    print(y_new)
    print(m.predict(x_new))
    print(accuracy_score(y_new, m.predict(x_new)))
    print(m.score(x_flat, y))