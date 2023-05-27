import pickle
import mne
from mne.preprocessing import corrmap, ICA
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
ch_refs = info['ch_refs']
sub_refs = info['sub_refs']
artifacts_labels = info['artifacts_labels']

def cleanbycorr(raws,icas,subs,chs,labels):

    for sub, ch, label in zip(subs,chs,labels):
        inds, _ = icas[sub].find_bads_eog(raws[sub], ch_name=ch)
        template = icas[sub].get_components()[:, inds[0]]
        corrmap(icas, template=template, threshold='auto', label=label, plot=False)

    reconst_raws = []
    for raw,ica in zip(raws,icas):
        flag = True
        for label in labels:
            if ica.labels_[label] != []:
                # ica.plot_components(picks=ica.labels_[label])
                ica.exclude = ica.labels_[label]
                # ica.plot_sources(raw, show_scrollbars=False)
                flag = False
        if flag:
            reconst_raws.append(raw)
            continue

        reconst_raw = raw.copy()  # ica.apply() changes the Raw object in-place, so let's make a copy first
        ica.apply(reconst_raw)

        reconst_raws.append(reconst_raw)

    return reconst_raws


reconst_raws = cleanbycorr(raws, icas, sub_refs, ch_refs, artifacts_labels)

data = []
for raw in reconst_raws:
    data.append(raw.get_data())
cleaned_data = np.dstack(data).transpose([2,0,1])

with open(f'./datafiles/{dbname}/cleanedData.pkl', 'wb') as f:
    pickle.dump(cleaned_data, f)
