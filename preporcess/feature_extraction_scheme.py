import pandas as pd
import numpy as np
from digital_processing import bp_filter, notch_filter, plot_signal
from feature_extraction import features_estimation

# Load data
# signal_path = 'raw_data/emg_signal.xlsx'
# signal_path = 'raw_data/multichannel_emg_signals.xlsx'
# emg_signal_multichannel = pd.read_excel(signal_path).values
sampling_frequency = 2e3
frame = 50
step = 10

# save_folder = 'f'+str(frame)+'\\'

s_range = [3,4,5]
d_range, m_range, t_range = 30, 22, 4
for s in s_range:
    for n0 in range(d_range):
        load_folder = 'S'+str(s)+'/day' + str(n0 + 1) + '/'
        for n1 in range(m_range):
            for n2 in range(t_range):
                filename = 'D' + str(n0 + 1) + 'M' + str(n1 + 1) + 'T' + str(n2 + 1)

                # _day = 9
                # filename = 'D' + str(_day) + 'M' + str(n1 + 1) + 'T' + str(n2 + 1)

                # signal_path = 'raw_data/'+filename+'.csv'
                signal_path = 'raw_data/' + load_folder + filename + '.csv'
                print("\nLoad from:", signal_path)
                emg_signal_multichannel = pd.read_csv(signal_path).values
                # emg_signal_multichannel = np.array(emg_signal_multichannel.tolist())
                frames, num_channels = np.shape(emg_signal_multichannel)
                print("emg_signal shape: ", np.shape(emg_signal_multichannel))
                print(frames, num_channels)

                result = []

                for channel_i in range(num_channels):
                    emg_signal = emg_signal_multichannel[:,channel_i]
                    channel_name = 'ch_'+str(channel_i)


                    # Plot raw sEMG signal
                    # plot_signal(emg_signal, sampling_frequency, channel_name)

                    # Biomedical Signal Processing
                    emg_signal = emg_signal.reshape((emg_signal.size,))
                    # filtered_signal = notch_filter(emg_signal, sampling_frequency,
                    #                                plot=False)
                    # filtered_signal = bp_filter(filtered_signal, 10, 500,
                    #                             sampling_frequency, plot=False)

                    filtered_signal = emg_signal

                    # EMG Feature Extraction
                    emg_features, features_names = features_estimation(filtered_signal, channel_name,
                                                                       sampling_frequency, frame, step)
                    emg_features = emg_features.to_numpy()
                    print(features_names)
                    print(np.shape(emg_features))
                    # result.append(emg_features)
                    for i in range(len(emg_features)):
                        # print("[", i, "] = ", emg_features[i])
                        result.append(emg_features[i])
                    # print(emg_features[0])
                print("\nResult shape: ", np.shape(np.array(result)))
                result = np.array(result)

                df = pd.DataFrame(result.T)
                filepath = 'extracted/S'+str(s)+'/day'+str(n0 + 1)+'/'+filename+'_extracted.xlsx'
                df.to_excel(filepath, index=False)
                print("Saved to:", filepath)