import os
import pickle
import numpy as np
from scipy import io
import json
import mne

with open('currentDB.json', 'r') as f:
    dbname = json.load(f)['dbname']

with open(f'./datafiles/{dbname}/info.json', 'r') as f:
    info = json.load(f)

dbpath = info['dbpath']
selected_channels = info['selected_channels']
ch_names = info['ch_names']
fs = info['fs']
tRec = info['tRec']

n = fs*tRec
n_channels = int(len(ch_names)+1)

data_list = []
label_list = []
for filename in os.listdir(dbpath):
    if filename.endswith('EC.edf'):
        file = f'{dbpath}/{filename}'
        data = mne.io.read_raw_edf(file)
        raw_data = data.get_data()

        if raw_data.shape[1] < n:
            continue

        selected_data = raw_data[:n_channels, :n]
        data_list.append(selected_data)
        # Verifica se o paciente é saudável (0) ou diagnosticado com depressão (1)
        if 'MDD' in filename:
            label_list.append(1)
        else:
            label_list.append(0)

data_array = np.array(data_list)

Cz_ref = data_array[:,17,:]
for subject in range(data_array.shape[0]):
    Cz_ref = data_array[subject, 17, :]
    data_array[subject,:-1,:] = data_array[subject,:-1,:] - Cz_ref


with open(f'./datafiles/{dbname}/pickledData.pkl', 'wb') as f:
    pickle.dump(data_array[:,:-1,:], f)

with open(f'./datafiles/{dbname}/dataLabels.pkl', 'wb') as f:
    pickle.dump(label_list, f)