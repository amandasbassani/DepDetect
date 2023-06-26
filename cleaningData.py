import pickle
import numpy as np
import json
import matplotlib.pyplot as plt

with open('currentDB.json', 'r') as f:
    dbname = json.load(f)['dbname']

with open(f'./datafiles/{dbname}/info.json', 'r') as f:
    info = json.load(f)

with open(f'./datafiles/{dbname}/rawedData.pkl', 'rb') as f:
    raws = pickle.load(f)

with open(f'./datafiles/{dbname}/componentsICA.pkl', 'rb') as f:
    icas = pickle.load(f)

fs = info['fs']
i = 0
reconst_raws = []
for raw, ica in zip(raws, icas):
    bad_indices1, _ = ica.find_bads_eog(raw, ch_name='Fp2')
    bad_indices1 = set(bad_indices1)
    bad_indices2, _ = ica.find_bads_eog(raw, ch_name='F8')
    bad_indices2 = set(bad_indices2)
    bad_indices3, _ = ica.find_bads_eog(raw, ch_name='T4')
    bad_indices3 = set(bad_indices3)
    bad_indices4, _ = ica.find_bads_eog(raw, ch_name='O1')
    bad_indices4 = set(bad_indices4)
    eog_indices = list(bad_indices1.union(bad_indices2).union(bad_indices3).union(bad_indices4))

    if eog_indices:
        ica.plot_sources(raw, picks=eog_indices, show_scrollbars=False)
        ica.plot_components(picks=eog_indices, colorbar=True)
        ica.exclude = eog_indices
    else:
        reconst_raws.append(raw)
        continue

    reconst_raw = raw.copy()
    ica.apply(reconst_raw)
    reconst_raws.append(reconst_raw)

data = []
for raw in reconst_raws:
    data.append(raw.get_data())
cleaned_data = np.dstack(data).transpose([2,0,1])

with open(f'./datafiles/{dbname}/cleanedData.pkl', 'wb') as f:
    pickle.dump(cleaned_data, f)
