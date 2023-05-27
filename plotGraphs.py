import pickle
import numpy as np
import matplotlib.pyplot as plt
import json

subject = 49
ch_name = 'T4'
filename1 = 'filteredData'
filename2 = 'cleanedData'

with open('currentDB.json', 'r') as f:
    dbname = json.load(f)['dbname']

with open(f'./datafiles/{dbname}/info.json', 'r') as f:
    info = json.load(f)

with open(f'./datafiles/{dbname}/{filename1}.pkl', 'rb') as f:
    data1 = pickle.load(f)

with open(f'./datafiles/{dbname}/{filename2}.pkl', 'rb') as f:
    data2 = pickle.load(f)

selected_channels = info['selected_channels']
ch_names = info['ch_names']
ch_types = info['ch_types']
tRec = info['tRec']
fs = info['fs']

n = fs*tRec
n_channels = len(ch_names)
t = np.linspace(0, n / fs, n, False)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all', sharey='all')
fig.suptitle(f'Subject {subject} Channel {ch_name}')
ax1.plot(t, data1[subject,ch_names.index(ch_name),:])
ax1.set_title(filename1)
ax2.plot(t, data2[subject,ch_names.index(ch_name),:])
ax2.set_title(filename2)
ax2.set_xlabel('Time [seconds]')
plt.show()
