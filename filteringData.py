import pickle
import numpy as np
from scipy.signal import butter, iirnotch, filtfilt, sosfilt
import json

with open('currentDB.json', 'r') as f:
    dbname = json.load(f)['dbname']

with open(f'./datafiles/{dbname}/info.json', 'r') as f:
    info = json.load(f)

selected_channels = info['selected_channels']
ch_names = info['ch_names']
tRec = info['tRec']
fs = info['fs']
f0 = info['f0']
fa = info['fa']
fb = info['fb']

n = fs*tRec
n_channels = len(ch_names)

with open(f'./datafiles/{dbname}/pickledData.pkl', 'rb') as f:
    data = pickle.load(f)

# Filtrando sinal, mantendo o  mesmo shape dos dados (subjects x channels x n_samples)
time_padding = 10  # seconds
n_padding = fs*time_padding
padding = data[:,:,:n_padding]
filtered_data = np.dstack((padding,data))

for subject in range(data.shape[0]):
    for channel in range(data.shape[1]):
        b, a = iirnotch(f0, 10, fs=fs)
        filtered_data[subject, channel, :] = filtfilt(b, a, filtered_data[subject, channel, :])
        sos = butter(4, [fa, fb], 'bp', fs=fs, output='sos')
        filtered_data[subject, channel, :] = sosfilt(sos, filtered_data[subject, channel, :])
filtered_data = filtered_data[:,:,n_padding:]

with open(f'./datafiles/{dbname}/filteredData.pkl', 'wb') as f:
    pickle.dump(filtered_data, f)
