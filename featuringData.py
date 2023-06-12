import pickle
import numpy as np
from scipy.signal import welch
import json

with open('currentDB.json', 'r') as f:
    dbname = json.load(f)['dbname']

with open(f'./datafiles/{dbname}/info.json', 'r') as f:
    info = json.load(f)

with open(f'./datafiles/{dbname}/cleanedData.pkl', 'rb') as f:
    pkdata = pickle.load(f)

fs = info['fs']
window_time = info['window_time']//1000

data = []
for i in range(0,pkdata.shape[2],window_time*fs):
    data.append(pkdata[:,:,i:i+window_time*fs])

data = np.vstack(data)

f, Pxx = welch(x=data,
               fs=float(fs),
               window='hamming',
               nperseg=100,
               axis=2)

featured_data = Pxx.transpose((0,2,1))
print(featured_data.shape)
print(f[0:21])

with open(f'./datafiles/{dbname}/featuredData.pkl', 'wb') as file:
    pickle.dump(featured_data, file)
