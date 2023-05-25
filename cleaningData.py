import pickle
import mne
from mne.preprocessing import corrmap
import matplotlib.pyplot as plt
import numpy as np
import json

with open('currentDB.json', 'r') as f:
    dbname = json.load(f)['dbname']

with open(f'./datafiles/{dbname}/info.json', 'r') as f:
    info = json.load(f)

with open(f'./datafiles/{dbname}/rawedData.pkl', 'rb') as f:
    raws = pickle.load(f)

with open(f'./datafiles/{dbname}/componentsICA.pkl', 'rb') as f:
    icas = pickle.load(f)

fs = info['fs']
ch_ref_eog = info['ch_ref_eog']
sub_ref_eog = info['sub_ref_eog']

# use the 27th subject as template; use E22 as proxy for EOG
eog_inds, _ = icas[sub_ref_eog].find_bads_eog(raws[sub_ref_eog], ch_name=ch_ref_eog)
template_eog_component1 = icas[sub_ref_eog].get_components()[:, eog_inds[0]]

corrmap(icas, template=template_eog_component1, threshold=0.8, label="eye_mov", plot=False)

data = []
for raw,ica in zip(raws,icas):
    if ica.labels_["eye_mov"] != []:
        # ica.plot_components(picks=ica.labels_["eye_mov"])
        ica.exclude = ica.labels_["eye_mov"]
        # ica.plot_sources(raw, show_scrollbars=False)
    else:
        data.append(raw.get_data())
        continue

    reconst_raw = raw.copy()  # ica.apply() changes the Raw object in-place, so let's make a copy first
    ica.apply(reconst_raw)

    data.append(reconst_raw.get_data())

cleaned_data = np.dstack(data).transpose([2,0,1])

with open(f'./datafiles/{dbname}/cleanedData.pkl', 'wb') as f:
    pickle.dump(cleaned_data, f)
