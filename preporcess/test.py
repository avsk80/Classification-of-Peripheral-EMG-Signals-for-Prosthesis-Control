import pandas as pd
import numpy as np
from oct2py import Oct2Py
from digital_processing import bp_filter, notch_filter, plot_signal
from feature_extraction import features_estimation

# Load data
# signal_path = 'raw_data/emg_signal.xlsx'
# signal_path = 'raw_data/multichannel_emg_signals.xlsx'
# emg_signal_multichannel = pd.read_excel(signal_path).values
sampling_frequency = 2e3
frame = 50
step = 10


def getrmsfeat(data, winsize=50, wininc=10):
    m = Oct2Py()
    return m.getrmsfeat(data, winsize, wininc)

def getmavfeat(data, winsize=50, wininc=10):
    m = Oct2Py()
    return m.getmavfeat(data, winsize, wininc)

def getzcfeat(data, winsize=50, wininc=10, deadzone=0.01):
    m = Oct2Py()
    return m.getzcfeat(data, deadzone, winsize, wininc)

def getsscfeat(data, winsize=50, wininc=10, deadzone=0.01):
    m = Oct2Py()
    return m.getsscfeat(data, deadzone, winsize, wininc)

def getwlfeat(data, winsize=50, wininc=10):
    m = Oct2Py()
    return m.getwlfeat(data, winsize, wininc)
# (29, 8)

def getarfeat(data, winsize=50, wininc=10, ar_order=6):
    m = Oct2Py()
    m.eval('pkg load signal')
    return m.getarfeat(data, ar_order, winsize, wininc)

def extract_feature(data, winsize=50, wininc=10):
    # print("data shape:", np.shape(data))
    feat1 = getrmsfeat(data, winsize, wininc)
    min_len = len(feat1)
    # print(np.shape(feat1))
    feat2 = getmavfeat(data, winsize, wininc)
    min_len = min(min_len, len(feat2))
    # print(np.shape(feat2))
    feat3 = getzcfeat(data, winsize, wininc)
    min_len = min(min_len, len(feat3))
    # print(np.shape(feat3))
    feat4 = getsscfeat(data, winsize, wininc)
    min_len = min(min_len, len(feat4))
    # print(np.shape(feat4))
    feat5 = getwlfeat(data, winsize, wininc)
    min_len = min(min_len, len(feat5))
    # print(np.shape(feat5))
    feat6 = getarfeat(data, winsize, wininc)
    min_len = min(min_len, len(feat6))
    # print(np.shape(feat6))

    result = feat1[:min_len]
    for f in (feat2, feat3, feat4, feat5, feat6):
        result = np.append(result, f[:min_len], axis=1)

    return result


d_range, m_range, t_range = 1, 1, 1
for n0 in range(d_range):
    load_folder = 'day' + str(n0 + 1) + '/'
    for n1 in range(m_range):
        for n2 in range(t_range):
            filename = 'D' + str(n0 + 1) + 'M' + str(n1 + 1) + 'T' + str(n2 + 1)
            signal_path = 'raw_data/' + load_folder + filename + '.csv'
            emg_signal_multichannel = pd.read_csv(signal_path).values


# a = np.array([[1,2],[3,4],[5,6]])
# b = np.array([[7],[8],[9]])
# print(np.append(a, b, axis=1))
# print('asdf')
feat = extract_feature(emg_signal_multichannel)
print(np.shape(feat))

# feat = getrmsfeat(emg_signal_multichannel, 50, 10)
# print(np.shape(feat))
# # print(len(feat))
# feat = getmavfeat(emg_signal_multichannel, 50, 10)
# print(np.shape(feat))
# feat = getzcfeat(emg_signal_multichannel, 50, 10)
# print(np.shape(feat))
# feat = getsscfeat(emg_signal_multichannel, 50, 10)
# print(np.shape(feat))
# feat = getwlfeat(emg_signal_multichannel, 50, 10)
# print(np.shape(feat))
# feat = getarfeat(emg_signal_multichannel, 50, 10)
# print(np.shape(feat))