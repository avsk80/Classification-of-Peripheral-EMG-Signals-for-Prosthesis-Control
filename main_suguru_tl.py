# from data import DATA
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
# from plot_keras_history import show_history, plot_history
import numpy as np
import keras
import tensorflow as tf
from keras.models import load_model, model_from_json
from sklearn.metrics import confusion_matrix
from test_suguru import test
from suguru_combine_t import combine_t, compile_training
from LDA_test_copy import test_LDA


load = False

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

def train_net(data_header=_data_header, load_model=load,
              load_train_data=False, test_only=False, validation_day=16,
              inner_model_name='', outer_model_name='', model_name=model_str,
              epoch=50, batch_size=200):
    es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1,
                                          patience=50,
                                          # min_delta=.05,
                                          start_from_epoch=10,
                                          restore_best_weights=True)

    if inner_model_name == '':
        inner_model_name = inner_model_str
    if outer_model_name == '':
        outer_model_name = outer_model_str


    if validation_day >= 0:
        manual_val = 1
    else:
        manual_val = validation_day
        validation_day = 16

    if not load_model:
        # data = DATA_DB6(load=True, data_header=_data_header, model_type=_model_type)
        data = DATA_SUGURU(load=load_train_data, data_header=data_header, model_type=_model_type,
                           frame=30, step=5, test_day=validation_day, test_only=test_only)
        print("data.input_shape =", data.input_shape, flush=True)
        print("data num_classes =", data.num_classes, flush=True)

        # x_train = np.concatenate((data.x_train, data.x_test))
        # y_train = np.concatenate((data.y_train, data.y_test))
        x_train = data.x_train
        y_train = data.y_train

        if _model_type == "lenet":
            model, inner_model, outer_model = lenet.create_lenet(input_shape=data.input_shape,
                                                                nb_classes=data.num_classes,
                                                                # inner_model_name=inner_model_name,
                                                                # outer_model_name=outer_model_name
                                                                 )
            # model, inner_model = lda.create_lda(input_shape=data.input_shape)

        elif _model_type == "rnn":
            x_train = x_train[:35000]
            y_train = y_train[:35000]
            model = rnn.create_rnn(nb_classes=data.num_classes, input_shape=(batch_size,300,16),
                                   inner_model_name=inner_model_name)
            # model = rnn.create_rnn(nb_classes=data.num_classes, input_shape=(batch_size,16,300))
            # x_train = x_train * 10000
        elif _model_type == "rnn2":
            print("rnn2")
            model, inner_model = rnn2.create_rnn(nb_classes=data.num_classes, input_shape=data.input_shape)
        elif _model_type == "fc":
            model = fc.create_fc(nb_classes=data.num_classes)

        model._name = _model_type + "_" + _data_header
        print(model.get_config())
        # model = lenet.add_prefix(model, 'temp_prefix')
        for layer in model.layers:
            print(layer.name)

        checkpoint_filepath = '/Models/checkpoint/'
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)

        # model.load_weights(checkpoint_filepath)
        print("x_train, y_train shape: ", np.shape(x_train), np.shape(y_train))


        # hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch, validation_split=0.20,
        #                  callbacks=[model_checkpoint_callback])

        if manual_val >= 0:
            hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch,
                             validation_data=(data.x_test, data.y_test),
                             callbacks=[es])
        else:
            if manual_val == -1:
                hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch,
                                 validation_split=0.05,
                                 callbacks=[es])
                # hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch,
                #                  validation_split=0.01,
                #                  callbacks=[model_checkpoint_callback])
            else:
                hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch,
                                 callbacks=[es])

        # model.save("Models/" + _model_type + "_" + _data_header + ".h5")
        model.save(model_str)
        if not model_str == model_name:
            model.save(model_name)
        inner_model.trainable = False
        inner_model.save(inner_model_name)
        if _model_type == "lenet":
            outer_model.save(outer_model_name)
        print("SAVING final model to:", model_str, inner_model_name)

        # show_history(hist)
        # plot_history(hist, path="standard.png")
        acc = model.evaluate(x_train, y_train, return_dict=True)
        val = model.evaluate(data.x_test, data.y_test, return_dict=True)

        return acc['accuracy'], val['accuracy']

    else:
        # data = DATA_DB6(load=True, data_header=_data_header)
        # model = load_model("Models/" + _model_type + "_" + _data_header + ".h5")
        model = load_model("Models/" + load_model_name + ".h5")
        return 0, 0



def test_main(rev=False, start_day=1, end_day=30, data_header='', load_model_name=load_model_name):
    t1 = time.perf_counter()
    result = []
    for i in range(start_day, end_day+1):
        tt1 = time.perf_counter()
        print("\n\nTest day %d:" % i, flush=True)
        result.append([i, test(i, path=data_header, load_model_name=load_model_name, model_type=_model_type)])
        tt2 = time.perf_counter()
        print("Day %d test duration: %d sec" % (i, (tt2 - tt1)))

    if rev:
        result.reverse()
    result = np.array(result)
    print("\n\n Result: \n", result)
    np.savetxt("result.csv", result.T, delimiter=",")
    t2 = time.perf_counter()
    print(" Time elapsed (min): %2f", (t2 - t1) / 60)
    return result.tolist()

def model_select_metric(acc, val):
    return val


if __name__ == '__main__':
    print("Running main...", flush=True)
    subject_list = [1,2,3,4,5]

    load_whole_model = False
    load_inner_model = False
    load_outer_model = False
    use_existing_combine_t = False
    load_train = False
    # first_training_day = 23
    # last_training_day = 23

    for training_day in range(1, 15):
    # for training_day in range(1, 2):  # Sets first training day to day 1, so total training days = 1-15
        # If you change this, need to set load_train = false for training data to update

        num_epochs = 1000   # uses early stopping, so more of an upper limit
        batch_size = 300

        # for training_day in range(15, 16):
        # first_training_day = training_day
        # last_training_day = training_day+13
        # val_day = training_day+14
        first_training_day = 1
        last_training_day = training_day
        val_day = training_day+1
        #     last_training_day = first_training_day+5
        additional_t_day = []
        num_models_best_of = 3

        test_day_start = val_day+1
        # test_day_start = first_training_day
        # test_day_end = test_day_start+7
        test_day_end = 30


        data_folder_offset = 29

        if not load_train:
            for subject_num in subject_list:
                data_folder_index = subject_num+data_folder_offset
                path = "Data/suguru/" + str(data_folder_index)
                # combine_t(read_path=path + '/', trials=[1, 2], day_range=(16, 30), save_day_offset=-15)
                # combine_t(read_path=path + '/', trials=[3, 4], day_range=(16, 30))
                if not use_existing_combine_t:
                    combine_t(read_path=path + '/', trials=[1,2,3,4], day_range=(1, 30))
                compile_training(save_to=str(data_folder_index), folder='Data/suguru/',
                                 first_training_day=first_training_day,
                                 last_training_day=last_training_day,
                                 additional_training_days=additional_t_day)
                data = DATA_SUGURU(load=False, data_header=path, model_type=_model_type,
                                   frame=30, step=5, test_day=25)

        model_acc = []
        for subject_num in subject_list:
            data_folder_index = subject_num+data_folder_offset

            # combine_t(read_path='Data/suguru/'+str(data_folder_index)+'/')
            # compile_training(save_to=str(data_folder_index), folder='Data/suguru/')

            acc = -1
            path = "Data/suguru/" + str(data_folder_index)
            t1 = time.perf_counter()

            if load_whole_model:
                model = load_model("Models/S" + str(subject_num) + "_" + load_model_name + ".h5")
            else:

                # data_folder_index = 112

                if _model_type == "lenet":
                    inner_model_name = "Models/S" + str(subject_num) + "_lenet_inner_test.h5"
                else:
                    inner_model_name = "Models/S" + str(subject_num) + "_rnn2_inner_test.h5"

                if _model_type == "lenet":
                    outer_model_name = "Models/S" + str(subject_num) + "_lenet_outer_test.h5"
                else:
                    outer_model_name = "Models/S" + str(subject_num) + "_rnn2_outer_test.h5"

                acc, val = train_net(data_header=path, load_train_data=False,
                                     validation_day=val_day,
                                     inner_model_name=inner_model_name,
                                     outer_model_name=outer_model_name,
                                     epoch=num_epochs, batch_size=batch_size)
                score = model_select_metric(acc, val)

                model = load_model("Models/" + load_model_name + ".h5")
                inner_model = load_model(inner_model_name)
                t2 = time.perf_counter()
                print(int((t2-t1)/60), "min to train 1st network")

                for n_models in range(max(num_models_best_of-1, 0)):
                    new_acc, new_val = train_net(data_header=path, load_train_data=True,
                                                 validation_day=val_day,
                                                 inner_model_name=inner_model_name,
                                                 outer_model_name=outer_model_name,
                                                 epoch=num_epochs, batch_size=batch_size)
                    new_score = model_select_metric(new_acc, new_val)
                    if new_score > score:
                        score = new_score
                        model = load_model("Models/" + load_model_name + ".h5")
                        inner_model = load_model(inner_model_name)

                print("Train: Saving model to", "Models/S" + str(subject_num) + "_" + load_model_name + ".h5")
                model.save("Models/S" + str(subject_num) + "_" + load_model_name + ".h5")
                inner_model.save(inner_model_name)
                model_acc.append(acc)

                t2 = time.perf_counter()
                print(int((t2-t1)/60), "min to train all networks")

                print("  Final accuracy: ", acc, flush=True)



        # Testing:
        for subject_num in subject_list:
            data_folder_index = subject_num+data_folder_offset
            path = "Data/suguru/" + str(data_folder_index)
            model = load_model("Models/S" + str(subject_num) + "_" + load_model_name + ".h5")

            try:
                acc = model_acc[subject_num-1]
            except:
                acc = -1

            result = np.array([[0, acc]] + test_main(data_header=path,
                                                     start_day=test_day_start,
                                                     end_day=test_day_end,
                                                     load_model_name="Models/S" + str(subject_num) + "_" + load_model_name + ".h5"))
            # np.savetxt('Results/result_' + str(data_folder_index) + '.csv', result.T, delimiter=",")
            # np.savetxt('Results/result_' + str(last_training_day - 1) + '.csv', result.T, delimiter=",")


            id_str = 'S' + str(subject_num) + '_Train-' + str(first_training_day) + '-' + str(max(val_day, last_training_day))
            np.savetxt('Results/main_non_tl_result_' + id_str + '.csv', result.T, delimiter=",")

            t2 = time.perf_counter()
            print("Total time elapsed for trial %d: %d min" % (data_folder_index, int((t2 - t1) / 60)))

        test_LDA(test_day_start=test_day_start, test_day_end=test_day_end)


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



