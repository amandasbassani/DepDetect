import os
import pickle

import numpy as np
from scipy import io

n = 75000  # numero de amostras de tempo

# Pasta contendo os arquivos .mat
folder_path = 'D:\\amand\\Documentos\\Datasets\\EEG_128channels_resting_lanzhou_2015-20230325T144701Z-002\\Arquivos .mat'
database_name = 'modmadata'

# Lista de canais desejados
selected_channels = [21, 8, 23, 123, 35, 103, 51, 91, 69, 82, 32, 121, 44, 107, 57, 95]
ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6']

data_list = []
label_list = []
for filename in os.listdir(folder_path):
    if filename.endswith('.mat'):
        mat_file = io.loadmat(os.path.join(folder_path, filename))
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
label_array = np.array(label_list)

with open(f'./datafiles/{filename}.pkl', 'wb') as f:
    pickle.dump(data_array, f)

with open(f'./datafiles/label_{filename}.pkl', 'wb') as f:
    pickle.dump(label_array, f)