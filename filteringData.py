import pickle
import numpy as np
from scipy.signal import butter, iirnotch, filtfilt, sosfilt

filename = 'modmadata'  # nome do arquivo pickle de formato (subject x samples x channels x movements)
n = 75000  # numero de amostras
fs = 250  # taxa de amostragem
f0 = 50  # frequencia da rede (na China é 50 Hz)

def filtering(data, f0, fs):
    print(data.shape)
    time_padding = 10  # seconds
    n_padding = fs*time_padding
    padding = data[:,:,:n_padding]
    filtered_data = np.dstack((padding,data))
    print(filtered_data.shape)
    for subject in range(data.shape[0]):
        for channel in range(data.shape[1]):
            b, a = iirnotch(f0, 5, fs=fs)
            filtered_data[subject, channel, :] = filtfilt(b, a, filtered_data[subject, channel, :])
            sos = butter(5, [0.5, 50], 'bp', fs=fs, output='sos')
            filtered_data[subject, channel, :] = sosfilt(sos, filtered_data[subject, channel, :])
    filtered_data = filtered_data[:,:,n_padding:]
    print(filtered_data.shape)
    return filtered_data


# Abrindo o arquivo pickle
with open(f'./datafiles/{filename}.pkl', 'rb') as f:
    data_array = pickle.load(f)

# Recuperando vetores de dados: o primeiro com shape (subjects x channels x n_samples)
data_array = data_array[:, :, 0:n]

# Filtrando sinal, mantendo o  mesmo shape dos dados (subjects x channels x n_samples)
data_filtered = filtering(data_array, f0, fs)

with open(f'./datafiles/filtered_{filename}.pkl', 'wb') as f:
    pickle.dump(data_filtered, f)
