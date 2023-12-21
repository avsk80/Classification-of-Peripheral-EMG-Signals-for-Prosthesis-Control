import pandas as pd
import numpy as np

from oct2py import Oct2Py
from digital_processing import bp_filter, notch_filter, plot_signal
from feature_extraction import features_estimation, extract_feature

# Load data
# signal_path = 'raw_data/emg_signal.xlsx'
# signal_path = 'raw_data/multichannel_emg_signals.xlsx'
# emg_signal_multichannel = pd.read_excel(signal_path).values
sampling_frequency = 2e3
frame = 50
step = 10

# save_folder = 'f'+str(frame)+'\\'
subject_list = [1,2,3,4,5]
d_range, m_range, t_range = 30, 22, 4
# d_range, m_range, t_range = 1, 1, 1
m = Oct2Py()
m.eval('pkg load signal')
for s in subject_list:
    s_folder = 'S'+str(s)+'/'
    for n0 in range(1,d_range+1):
        working_folder = s_folder + 'day' + str(n0) + '/'
        for n1 in range(1,m_range+1):
            for n2 in range(1,t_range+1):
                filename = 'D' + str(n0) + 'M' + str(n1) + 'T' + str(n2)

                # _day = 9
                # filename = 'D' + str(_day) + 'M' + str(n1) + 'T' + str(n2)

                # signal_path = 'raw_data/'+filename+'.csv'
                signal_path = 'raw_data/' + working_folder + filename + '.csv'
                print("\nLoad from:", signal_path)
                emg_signal_multichannel = pd.read_csv(signal_path).values
                # emg_signal_multichannel = np.array(emg_signal_multichannel.tolist())
                frames, num_channels = np.shape(emg_signal_multichannel)
                print("emg_signal shape: ", np.shape(emg_signal_multichannel))
                print(frames, num_channels)

                result = extract_feature(emg_signal_multichannel, m, frame, step)

                # for channel_i in range(num_channels):
                #     emg_signal = emg_signal_multichannel[:,channel_i]
                #     channel_name = 'ch_'+str(channel_i)
                #
                #     print(np.shape(np.array(emg_signal)))
                #     # e = extract_feature(emg_signal)
                #     # print(np.shape(e))
                #     # print(e)
                #
                #
                #
                #     # Plot raw sEMG signal
                #     # plot_signal(emg_signal, sampling_frequency, channel_name)
                #
                #     # Biomedical Signal Processing
                #     emg_signal = emg_signal.reshape((emg_signal.size,))
                #     filtered_signal = notch_filter(emg_signal, sampling_frequency,
                #                                    plot=False)
                #     filtered_signal = bp_filter(filtered_signal, 10, 500,
                #                                 sampling_frequency, plot=False)
                #
                #     print("filtered signal shape =", np.shape(filtered_signal))
                #     #
                #     # # EMG Feature Extraction
                #     # emg_features, features_names = features_estimation(filtered_signal, channel_name,
                #     #                                                    sampling_frequency, frame, step)
                #     # emg_features = emg_features.to_numpy()
                #     # print(features_names)
                #     # print(np.shape(emg_features))
                #     # # result.append(emg_features)
                #     # for i in range(len(emg_features)):
                #     #     # print("[", i, "] = ", emg_features[i])
                #     #     result.append(emg_features[i])
                #     # print(emg_features[0])
                print("\nResult shape: ", np.shape(np.array(result)))
                result = np.array(result)

                df = pd.DataFrame(result.T)
                filepath = 'extracted/' + working_folder + filename + '_extracted.xlsx'
                df.to_excel(filepath, index=False)
                print("Saved to:", filepath)
m.exit()