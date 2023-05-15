import mne
import pickle
from autoreject import get_rejection_threshold
import numpy as np

filename = 'filtered_modmadata'
n = 75000  # numero de amostras
# Define channel names and types
ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6']
ch_types = 'eeg'
fs = 250

# Abrindo o arquivo pickle
with open(f'./datafiles/{filename}.pkl', 'rb') as f:
    pkdata = pickle.load(f)

data_array = pkdata[:, :, 0:n]

data_preprocessed = []
for subject in range(data_array.shape[0]):
    data = data_array[subject, :, :]
    mne.set_log_level('error')  # reduce extraneous MNE output

    # Cria um raw data

    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=fs)
    raw = mne.io.RawArray(data, info)

    # Filtros pro ICA
    ica_low_cut = 1.0  # For ICA, we filter out more low-frequency power
    hi_cut = 30

    raw_ica = raw.copy().filter(ica_low_cut, hi_cut)

    # Break raw data into 1 s epochs
    tstep = 1.0
    events_ica = mne.make_fixed_length_events(raw_ica, duration=tstep)
    epochs_ica = mne.Epochs(raw_ica, events_ica,
                            tmin=0.0, tmax=tstep,
                            baseline=None,
                            preload=True)

    reject = get_rejection_threshold(epochs_ica)

    # ICA parameters
    random_state = 42  # ensures ICA is reproducable each time it's run
    ica_n_components = 0.99999  # Specify n_components as a decimal to set % explained variance

    # Fit ICA
    ica = mne.preprocessing.ICA(n_components=ica_n_components, random_state=random_state)

    ica.fit(epochs_ica, reject=reject, tstep=tstep)

    ica_z_thresh = 1.96
    eog_indices, _ = ica.find_bads_eog(raw_ica, ch_name=['Fp1', 'F8'], threshold=ica_z_thresh)
    ica.exclude = eog_indices

    raw_copy = raw.copy()
    ica.apply(raw_copy)
    data_array_preprocessed = raw_copy.get_data()

    data_preprocessed.append(data_array_preprocessed)

data_preprocessed = np.array(data_preprocessed).reshape(data_array.shape)


with open(f'./datafiles/preprocessed_{filename}.pkl', 'wb') as f:
    pickle.dump(data_processed, f)
