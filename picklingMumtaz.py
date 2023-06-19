import os
import pickle
import numpy as np
import json
import mne

dbname = 'mumtaz'

with open(f'./datafiles/{dbname}/info.json', 'r') as f:
    info = json.load(f)

dbpath = info['dbpath']
selected_channels = info['selected_channels']
ch_names = info['ch_names']
fs = info['fs']
tRec = info['tRec']

data_list = []
label_list = []
i = 0
for filename in os.listdir(dbpath):
    if filename.endswith('EC.edf'):
        file = f'{dbpath}/{filename}'
        data = mne.io.read_raw_edf(file)
        data = data.copy().pick_channels(selected_channels)
        raw_data = data.get_data()
        print(data.info)

        if raw_data.shape[1] < fs*tRec:
            continue

        selected_data = raw_data[:, :fs*tRec]
        data_list.append(selected_data)

        # Verifica se o paciente é saudável (0) ou diagnosticado com depressão (1)
        if 'MDD' in filename:
            label_list.append(1)
        else:
            label_list.append(0)
data_array = np.array(data_list)

for subject in range(data_array.shape[0]):
    Cz_ref = data_array[subject, -1, :]
    data_array[subject,:-1,:] = data_array[subject,:-1,:] - Cz_ref
data_array = data_array[:,:-1,:]

with open(f'./datafiles/{dbname}/pickledData.pkl', 'wb') as f:
    pickle.dump(data_array, f)

with open(f'./datafiles/{dbname}/dataLabels.pkl', 'wb') as f:
    pickle.dump(label_list, f)
