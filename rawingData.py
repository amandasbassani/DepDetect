import pickle
import mne
from mne.preprocessing import ICA
import json

with open('currentDB.json', 'r') as f:
    dbname = json.load(f)['dbname']

with open(f'./datafiles/{dbname}/info.json', 'r') as f:
    info = json.load(f)

selected_channels = info['selected_channels']
ch_names = info['ch_names']
tRec = info['tRec']
fs = info['fs']

n_channels = len(ch_names)

layout = mne.channels.make_standard_montage('standard_1020')
info = mne.create_info(ch_names=ch_names, ch_types='eeg', sfreq=fs)

with open(f'./datafiles/{dbname}/filteredData.pkl', 'rb') as f:
    data_array = pickle.load(f)

raws = []
icas = []
for subject in range(0,data_array.shape[0]):

    data = data_array[subject, :, :]
    raw = mne.io.RawArray(data, info)
    raw.set_montage(layout)

    filt_raw = raw.copy().load_data().filter(l_freq=1.0, h_freq=None)

    ica = ICA(n_components=16, max_iter=300, method="fastica")
    ica.fit(filt_raw, verbose="error")

    raws.append(raw)
    icas.append(ica)

with open(f'./datafiles/{dbname}/rawedData.pkl', 'wb') as f:
    pickle.dump(raws, f)

with open(f'./datafiles/{dbname}/componentsICA.pkl', 'wb') as f:
    pickle.dump(icas, f)
