# from data import DATA
# from data import DATA_MAT
# from data import DATA_DB6
from data import DATA_SUGURU
import data
# import lenet
# import rnn
# import rnn2
# import fc
import time
# from plot import plot_loss
# import matplotlib.pyplot as plt
# from plot_keras_history import show_history, plot_history
import numpy as np
# from sklearn.metrics import accuracy_score, confusion_matrix


# import keras
# import tensorflow as tf
from keras.models import load_model, model_from_json
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from test_suguru import test
from suguru_combine_t import combine_t, compile_training
# from LDA_model import suguru_LDA

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
# from sklearn.inspection import DecisionBoundaryDisplay
# from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    LinearDiscriminantAnalysis()
]

load = False
# 1 => .77
clf = classifiers[1]
# clf = suguru_LDA()

# input_shape_ = (28, 28, 1)
# nb_classes_ = 7
# input_shape_ = (16, data.num_frames_per_datapoint, 1)

input_shape_ = (252, data.num_frames_per_datapoint, 1)
nb_classes_ = 22


# process data
subject_num = 1
day = 1
test_day = 8

# Available models: lenet, fc, rnn2
_model_type = "lenet"

load_model_name = _model_type + '_test'
_data_header = "S"+str(subject_num)+"_D"+str(day)
_test_data_header = "S"+str(subject_num)+"_D"+str(test_day)
model_str = "Models/" + _model_type + "_test.h5"
inner_model_str = "Models/" + _model_type + "_inner_test.h5"
outer_model_str = "Models/" + _model_type + "_outer_test.h5"

def revert_one_hot(y_oh):
    y_oh = np.array(y_oh)
    result = []
    for sample in y_oh:
        result.append(sample.argmax(axis=0))
    return np.array(result)

def test_LDA(test_day_start=16, test_day_end=30):
    print("Running test_LDA...", flush=True)
    # combine_t(trials=[1, 2], day_range=(16, 30), save_day_offset=-15)
    # combine_t(trials=[3, 4], day_range=(16, 30))
    subject_list = [1,2,3,4,5]
    load_whole_model = False
    load_inner_model = True

    use_compiled_train = True
    load_train = False

    # test_day_start = 16
    # test_day_end = 30


    lda_solver = 'svd'

    data_folder_offset = 34
    if not use_compiled_train:
        for subject_num in subject_list:
            data_folder_index = subject_num + data_folder_offset
            print("\nCombining days 1-15...\n")
            combine_t(read_path='Data/suguru/' + str(data_folder_index) + '/',
                      trials=[1, 2], day_range=(16, 30), save_day_offset=-15)

            print("\nCombining days 16-30...\n")
            combine_t(read_path='Data/suguru/' + str(data_folder_index) + '/',
                      trials=[3, 4], day_range=(16, 30))
            # for data_folder_index in range(1, 5):


    total_acc = []
    for subject_num in subject_list:
        data_folder_index = subject_num + data_folder_offset


        acc = -1
        path = "Data/suguru/" + str(data_folder_index)
        t1 = time.perf_counter()



        if load_inner_model:
            if _model_type == "lenet":
                inner_model_name = "Models/S" + str(subject_num) + "_lenet_inner_test.h5"
                # inner_model_name = "Models/lenet_inner_test.h5"
            else:
                inner_model_name = "Models/S" + str(subject_num) + "_rnn2_inner_test.h5"
        else:
            inner_model_name = ''

        print("Loading inner model from", inner_model_name)
        inner_model = load_model(inner_model_name)

        lda_baseline = []
        result = []
        linearity = []
        total_linearity_list = []
        adj_total_linearity_list = []
        c_mat = np.zeros((22, 22))


        for day in range(test_day_start, test_day_end+1):
            print(path)
            data_1 = DATA_SUGURU(load=load_train, data_header=path, model_type=_model_type,
                               frame=30, step=5, test_day=day-15, save_to_day=day-15, test_only=True)
            (x1, y1) = (data_1.x_test, revert_one_hot(data_1.y_test))
            x1_flat = np.array([np.array(s).flatten() for s in x1])

            data_2 = DATA_SUGURU(load=load_train, data_header=path, model_type=_model_type,
                               frame=30, step=5, test_day=day, save_to_day=day, test_only=True)
            (x2, y2) = (data_2.x_test, revert_one_hot(data_2.y_test))
            x2_flat = np.array([np.array(s).flatten() for s in x2])

            print("x1 flatten shape:", np.shape(x1_flat))
            x = np.append(x1, x2, axis=0)
            x_flat = np.array([np.array(s).flatten() for s in x])
            y = np.append(y1, y2)
            print("x, y shapes:", np.shape(x), np.shape(y))
            # print("y min, max =", np.min(y), np.max(y))

            # Raw total linearity
            lda = LinearDiscriminantAnalysis(solver=lda_solver)
            # clf = suguru_LDA(solver=lda_solver)

            clf.fit(x_flat, y)
            total_linearity = clf.score(x_flat, y)

            # Adjusted total linearity
            inner_model_output = inner_model.predict(x)
            clf.fit(inner_model_output, y)

            adj_total_linearity = clf.score(inner_model_output, y)

            print("total linearity =", total_linearity)
            total_linearity_list.append([day, total_linearity, adj_total_linearity])
            print("adj total linearity =", adj_total_linearity)
            adj_total_linearity_list.append([day, adj_total_linearity])

            # print("x1:\n", x1.flatten()[:5])
            # print("x2:\n", x2.flatten()[:5])

            # LDA baseline

            lda.fit(x1_flat, y1)
            lda_acc1 = lda.score(x2_flat, y2)

            # clf = LinearDiscriminantAnalysis(solver=lda_solver)
            lda.fit(x2_flat, y2)
            lda_acc2 = lda.score(x1_flat, y1)

            # available options: svd, lsqr, eigen

            # train classifier 1
            inner_model_output = inner_model.predict(x1)
            clf.fit(inner_model_output, y1)

            linearity_score_1 = clf.score(inner_model_output, y1)

            # test classifier 1
            inner_model_output = inner_model.predict(x2)
            acc1 = clf.score(inner_model_output, y2)

            c_mat1 = confusion_matrix(y2, clf.predict(inner_model_output))

            # train classifier 2
            inner_model_output = inner_model.predict(x2)
            clf.fit(inner_model_output, y2)

            linearity_score_2 = clf.score(inner_model_output, y2)

            # test classifier 2
            inner_model_output = inner_model.predict(x1)
            acc2 = clf.score(inner_model_output, y1)

            c_mat2 = confusion_matrix(y1, clf.predict(inner_model_output))
            c_mat += c_mat1+c_mat2

            print(" Day", day, "accuracy:", [acc1, acc2],
                  ", percent linearly separable (linearity score):", [linearity_score_1, linearity_score_2], "\n")
            print(" LDA baseline:", [lda_acc1, lda_acc2])

            linearity.append((linearity_score_2+linearity_score_1)/2)
            result.append([day, (acc1+acc2)/2])
            lda_baseline.append([day, (lda_acc1+lda_acc2)/2])

        result, linearity, total_linearity_list = \
            np.array(result), np.array(linearity), np.array(total_linearity_list)
        lda_baseline = np.array(lda_baseline)
        print("\n\n Results for subject", subject_num, "\n",result)
        avg_acc = np.average(result.T[1])
        print("Average accuracy =", avg_acc)
        print("Average LDA accuracy =", np.average(lda_baseline.T[1]))
        print("Average linearity score =", np.average(linearity))
        print("Total linearity:")
        print(total_linearity_list)

        total_acc.append(avg_acc)

        id_str = 'S'+str(subject_num) + '_Test-' + str(test_day_start)+'-'+str(test_day_end)

        np.savetxt('Results/conf_mat_' + id_str + '.csv', c_mat, delimiter=",")
        np.savetxt('Results/TL_Clf_result_' + id_str + '.csv', result.T, delimiter=",")
        np.savetxt('Results/LDA_baseline_' + id_str + '.csv', lda_baseline.T, delimiter=",")

        # plt.rcParams["figure.figsize"] = [7.50, 3.50]
        # plt.rcParams["figure.autolayout"] = True
        #
        # x = np.array(result.T[0])
        # y = np.array(result.T[1])
        #
        # plt.title("Line graph")
        # plt.plot(x, y, color="red")
        #
        # plt.show()

    print("Overall accuracy across selected subjects =", np.average(total_acc))


if __name__ == '__main__':
    test_LDA()

