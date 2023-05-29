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
thresholds = info['thresholds']

def cleanbycorr(raws,icas,subs,chs,labels,thresholds):

    for sub, ch, label, threshold in zip(subs,chs,labels,thresholds):
        if ch == "F8" and label == "h_mov1":
            template = icas[sub].get_components()[:, 0]
        else:
            inds, _ = icas[sub].find_bads_eog(raws[sub], ch_name=ch)
            template = icas[sub].get_components()[:, inds[0]]
        corrmap(icas, template=template, threshold=threshold, label=label, plot=False)

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

def cleanbyproxy(raws, icas):
    reconst_raws = []
    for raw,ica in zip(raws,icas):
        bad_indices1, _ = ica.find_bads_eog(raw, ch_name='Fp2')
        bad_indices1 = set(bad_indices1)
        bad_indices2, _ = ica.find_bads_eog(raw, ch_name='F8')
        bad_indices2 = set(bad_indices2)
        bad_indices3, _ = ica.find_bads_eog(raw, ch_name='T4')
        bad_indices3 = set(bad_indices3)
        bad_indices4, _ = ica.find_bads_eog(raw, ch_name='O1')
        bad_indices4 = set(bad_indices3)
        eog_indices = list(bad_indices1.union(bad_indices2).union(bad_indices3).union(bad_indices4))

        if eog_indices != []:
            ica.exclude = eog_indices
        else:
            reconst_raws.append(raw)
            continue
        reconst_raw = raw.copy()
        ica.apply(reconst_raw)
        reconst_raws.append(reconst_raw)
    return reconst_raws

# reconst_raws = cleanbycorr(raws, icas, sub_refs, ch_refs, artifacts_labels, thresholds)
reconst_raws = cleanbyproxy(raws, icas)

data = []
for raw in reconst_raws:
    data.append(raw.get_data())
cleaned_data = np.dstack(data).transpose([2,0,1])

with open(f'./datafiles/{dbname}/cleanedData.pkl', 'wb') as f:
    pickle.dump(cleaned_data, f)
