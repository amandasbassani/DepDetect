import pickle
import matplotlib.pyplot as plt
import numpy as np
import json

filename = 'cleanedData'

with open('currentDB.json', 'r') as f:
    dbname = json.load(f)['dbname']

with open(f'./datafiles/{dbname}/info.json', 'r') as f:
    info = json.load(f)

with open(f'./datafiles/{dbname}/{filename}.pkl', 'rb') as f:
    data_array = pickle.load(f)

selected_channels = info['selected_channels']
ch_names = info['ch_names']
ch_types = info['ch_types']
tRec = info['tRec']
fs = info['fs']

n = fs*tRec
n_channels = len(ch_names)
t = np.linspace(0, n / fs, n, False)

for subject in range(data_array.shape[0]):
    fig, ax = plt.subplots(n_channels, 1, sharex='all', sharey='all', figsize=(30, 15))
    fig.suptitle(f'Subject {subject}')
    for i in range(n_channels):
        ax[i].plot(t, data_array[subject, i, :])
        ax[i].set_title(ch_names[i])
        ax[i].set_ylabel('Amplitude [uV]')
    ax[i].set_xlabel('Time [seconds]')

    plt.tight_layout()

    plt.savefig(f'./datafiles/{dbname}/images/{filename}S{subject}.png')
    print(f'Imagem {subject} salva!')
    plt.close()
