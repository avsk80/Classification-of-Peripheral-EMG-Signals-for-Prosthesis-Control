from data import DATA
from data import DATA_MAT
from data import DATA_DB6
import data
import lenet
import time
# import rnn
# import fc
# from plot import plot_loss
import matplotlib.pyplot as plt
from plot_keras_history import show_history, plot_history
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from keras.models import load_model, model_from_json
from sklearn.metrics import confusion_matrix
import collections


load = True
load_model_name = 'lenet_S5_D5'

# input_shape_ = (28, 28, 1)
nb_classes_ = 7
input_shape_ = (16, data.num_frames_per_datapoint, 1)
nb_classes_ = 5


# process data
subject_num = 5
day = 5
test_day = 1

# Available models: lenet, fc, rnn
_model_type = "lenet"
_data_header = "S"+str(subject_num)+"_D"+str(day)
_test_data_header = "S"+str(subject_num)+"_D"+str(test_day)


def combine_t(read_path='Data/suguru/0/', trials=[1, 2, 3, 4], day_range=(1, 30), save_day_offset=0):

    (day_start, day_end) = day_range
    # read_path = 'Data/suguru/'
    # read_path = ''

    for day in range(day_start, day_end+1):
        for m in range(1, 23):
            all_trials = []
            for t in trials:
                # filename = 'D'+str(day)+'M'+str(m)+'T'+str(t)+'.csv'
                # WS = pd.read_csv('Data/suguru/'+'day'+str(day)+'/'+filename)
                filename = 'D' + str(day) + 'M' + str(m) + 'T' + str(t)
                try:
                    WS = pd.read_excel(read_path+'day'+str(day)+'/'+filename+'_extracted.xlsx', engine='openpyxl')
                except:
                    WS = pd.read_csv(read_path+'day'+str(day)+'/'+filename + '.csv')
                WS_np = np.array(WS)
                print(np.shape(WS_np))
                all_trials.extend(WS_np.tolist())



            all_trials = np.array(all_trials)
            print(np.shape(all_trials))
            df = pd.DataFrame(all_trials)

            ## save to xlsx file

            filepath = read_path+'day'+str(day+save_day_offset)+'/M'+str(m)+'.xlsx'
            print("Saved to ", filepath)
            df.to_excel(filepath, index=False)

def remove_cols(np_matrix_2d, index_list):
    num_channels = 8
    rows, cols = np.shape(np_matrix_2d)
    num_features = int(cols/num_channels)

    if len(index_list) == 0:
        return np_matrix_2d
    if num_features*num_channels != cols:
        print("ERROR (remove_col): input matrix does not have 8 channels")
        return np_matrix_2d

    index_list.sort(reverse=True)
    index = index_list[0]
    if index>=num_features:
        print("ERROR (remove_col): invalid delete index")
        return np_matrix_2d

    result = np_matrix_2d

    for index in index_list:
        for i in reversed(range(num_channels)):
            result = np.delete(result, index+(i*num_features), 1)
        num_features-=1
    return result

def compile_ablation_data(list_remove_indices=[], save_to=''):
    list_remove_indices.sort(reverse=True)
    if save_to=='':
        save_to = str(list_remove_indices[0])

    if len(list_remove_indices) >= 0:
        for day in range(1, 31):
            for m in range(1, 23):
        # for day in range(1, 2):
        #     for m in range(1, 2):
                filename = 'day' + str(day) + '/M' + str(m) + '.xlsx'
                WS = pd.read_excel(filename, engine='openpyxl')
                WS_np = np.array(WS)
                WS_np = remove_cols(np.array(WS_np), list_remove_indices)

                # Normalize by column
                WS_np = WS_np.T
                for j in range(len(WS_np)):
                    # print("Before:")
                    # print(WS_np[j][:10])
                    # WS_np[j] =
                    WS_np[j] -= np.average(WS_np[j])
                    norm = np.linalg.norm(WS_np[j])
                    if norm != 0:
                        WS_np[j] /= norm
                    # print("After:")
                    # print(WS_np[j][:10])
                    # print("\n\n")
                WS_np = np.array(WS_np).T

                filepath = 'ablation/' + save_to + '/day' + str(day) + '/M' + str(m) + '.xlsx'
                print("Saved ablation data to", filepath)
                df = pd.DataFrame(WS_np)
                df.to_excel(filepath, index=False)

def compile_training(list_remove_indices=[], save_to='', folder='ablation/',
                     first_training_day=1,
                     last_training_day=15,
                     additional_training_days=[],
                     extend=False):
    # Compile training data
    # Currently only works for single index remove due to filepath
    num_channels = 8
    if save_to=='':
        if len(list_remove_indices) > 0:
            save_to = str(list_remove_indices[0])
        else:
            save_to = 'temp'

    for m in range(1, 23):
        all_trials = []
        for day in range(first_training_day, last_training_day+1):
        # for day in range(training_days, 31):
            filename = folder + save_to + '/day' + str(day) + '/M' + str(m) + '.xlsx'
            # print("folder", folder)
            # print("save_to", save_to)
            print("  Reading from", filename)
            WS = pd.read_excel(filename, engine='openpyxl')
            WS_np = np.array(WS)
            # print(np.shape(WS_np))
            all_trials.extend(WS_np.tolist())
        if additional_training_days != []:
            for day in additional_training_days:
                filename = folder + save_to + '/day' + str(day) + '/M' + str(m) + '.xlsx'
                # print("folder", folder)
                # print("save_to", save_to)
                print("  Reading from", filename)
                WS = pd.read_excel(filename, engine='openpyxl')
                WS_np = np.array(WS)
                # print(np.shape(WS_np))
                all_trials.extend(WS_np.tolist())



        # print(np.shape(all_trials[0]))
        df = pd.DataFrame(all_trials)
        # filepath = 'ablation/day'+str(day)+'/M' + str(m) + '.xlsx'
        # filepath = 'Data/suguru/M' + str(m) + '.xlsx'
        filepath = folder + save_to +'/M' + str(m) + '.xlsx'
        print("Saved training data to ", filepath, "\n")
        df.to_excel(filepath, index=False)



def main():

    # for i in file_list:
    #     compile_ablation_data([i], save_to=save_to_folder)
    #     compile_training([i], save_to=save_to_folder)


    # save_to_folder = '404-normalized'
    # delete_index_list = []
    #
    # t1 = time.time()
    # compile_ablation_data(delete_index_list, save_to=save_to_folder)
    # t2 = time.time()
    # print("\nTime elapsed for %d compile_ablation: %d min\n\n" % (1, int((t2 - t1) / 60)))
    # compile_training(delete_index_list, save_to=save_to_folder)
    # t2 = time.time()
    # print("\nTotal time elapsed for %d: %d min\n\n" % (1, int((t2 - t1) / 60)))

    # save_to_folder = 'acc-4-normalized'
    # delete_index_list = [2, 15, 8, 5]
    #
    # t1 = time.time()
    # compile_ablation_data(delete_index_list, save_to=save_to_folder)
    # t2 = time.time()
    # print("\nTime elapsed for %d compile_ablation: %d min\n\n" % (1, int((t2 - t1) / 60)))
    # compile_training(delete_index_list, save_to=save_to_folder)
    # t2 = time.time()
    # print("\nTotal time elapsed for %d: %d min\n\n" % (1, int((t2 - t1) / 60)))
    #
    # save_to_folder = 'acc-8-normalized'
    # delete_index_list = [2, 15, 8, 5, 1, 6, 7, 3]
    #
    # t1 = time.time()
    # compile_ablation_data(delete_index_list, save_to=save_to_folder)
    # t2 = time.time()
    # print("\nTime elapsed for %d compile_ablation: %d min\n\n" % (1, int((t2 - t1) / 60)))
    # compile_training(delete_index_list, save_to=save_to_folder)
    # t2 = time.time()
    # print("\nTotal time elapsed for %d: %d min\n\n" % (1, int((t2 - t1) / 60)))
    #
    #
    #
    # save_to_folder = 'degrade-4-normalized'
    # delete_index_list = [16, 6, 13, 0]
    #
    # t1 = time.time()
    # compile_ablation_data(delete_index_list, save_to=save_to_folder)
    # t2 = time.time()
    # print("\nTime elapsed for %d compile_ablation: %d min\n\n" % (1, int((t2 - t1) / 60)))
    # compile_training(delete_index_list, save_to=save_to_folder)
    # t2 = time.time()
    # print("\nTotal time elapsed for %d: %d min\n\n" % (1, int((t2 - t1) / 60)))
    #
    # save_to_folder = 'degrade-8-normalized'
    # delete_index_list = [16, 6, 13, 0, 5, 3, 12, 14]
    #
    # t1 = time.time()
    # compile_ablation_data(delete_index_list, save_to=save_to_folder)
    # t2 = time.time()
    # print("\nTime elapsed for %d compile_ablation: %d min\n\n" % (1, int((t2 - t1) / 60)))
    # compile_training(delete_index_list, save_to=save_to_folder)
    # t2 = time.time()
    # print("\nTotal time elapsed for %d: %d min\n\n" % (1, int((t2 - t1) / 60)))

    for i in range(10, 14):
        t1 = time.time()
        # compile_ablation_data([i])
        # t2 = time.time()
        # print("\nTime elapsed for %d compile_ablation: %d min\n\n" %(i, int((t2-t1)/60)))
        compile_training([i], save_to=str(i))
        t2 = time.time()
        print("\nTotal time elapsed for %d: %d min\n\n" %(i, int((t2-t1)/60)))

if __name__ == '__main__':
    # combine_t()
    main()