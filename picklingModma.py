import os
import pickle
import numpy as np
from scipy import io
import json

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

data_list = []
label_list = []
for filename in os.listdir(dbpath):
    if filename.endswith('.mat'):
        mat_file = io.loadmat(os.path.join(dbpath, filename))
        for var_name in mat_file:
            if "__" in var_name:
                continue
            var_shape = np.array(mat_file[var_name]).shape
            if var_shape[1] >= n:
                eeg_data = mat_file[var_name]
                selected_data = eeg_data[selected_channels, :n]
                data_list.append(selected_data)

                # Verifica se o paciente é saudável (0) ou diagnosticado com depressão (1)
                if '0201' in filename:
                    label_list.append(1)
                else:
                    label_list.append(0)

data_array = np.array(data_list)
print(data_array)
with open(f'./datafiles/{dbname}/pickledData.pkl', 'wb') as f:
    pickle.dump(data_array, f)

with open(f'./datafiles/{dbname}/dataLabels.pkl', 'wb') as f:
    pickle.dump(label_list, f)
