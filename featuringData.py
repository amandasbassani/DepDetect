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
bands = info['bands']
window_time = info['window_time']//1000

data = []
for i in range(0,pkdata.shape[2],window_time*fs):
    data.append(pkdata[:,:,i:i+window_time*fs])

data = np.vstack(data)

f, Pxx = welch(x=data,
               fs=float(fs),
               window='hamming',
               nperseg=100,
               # noverlap=0,
               axis=2)

# feature_vector = []
# for band in bands:
#     M = np.zeros((Pxx.shape[2]))
#     if band == 'delta':
#         M[f <= 5] = 1
#     elif band == 'theta':
#         M[5 < f] = 1
#         M[f <= 8] = 0
#     elif band == 'alpha':
#         M[8 < f] = 1
#         M[f <= 12] = 0
#     elif band == 'beta':
#         M[12 < f] = 1
#         M[f <= 30] = 0
#     elif band == 'gamma':
#         M[30 < f] = 1
#         M[f <= 45] = 0
#
#     feature_vector.append(np.sum((M*Pxx),axis=2)/np.count_nonzero(M))
#
# feature_vector = np.dstack(feature_vector)
# featured_data = feature_vector.transpose((0,2,1))
# print(featured_data.shape)

featured_data = Pxx.transpose((0,2,1))# [:,0:21,:]
print(featured_data.shape)
print(f[0:21])

with open(f'./datafiles/{dbname}/featuredData.pkl', 'wb') as file:
    pickle.dump(featured_data, file)
