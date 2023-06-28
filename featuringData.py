import pickle
import numpy as np
from scipy.signal import welch
import json
import matplotlib.pyplot as plt

with open('currentDB.json', 'r') as f:
    dbname = json.load(f)['dbname']

with open(f'./datafiles/{dbname}/info.json', 'r') as f:
    info = json.load(f)

with open(f'./datafiles/{dbname}/cleanedData.pkl', 'rb') as f:
    pkdata = pickle.load(f)

fs = info['fs']
window_time = info['window_time']

data = []
for i in range(0,pkdata.shape[2],window_time*fs):
    data.append(pkdata[:,:,i:i+window_time*fs])

data = np.vstack(data) * 1e6

f, Pxx = welch(x=data,
               fs=float(fs),
               window='hamming',
               nperseg=100,
               axis=2)

featured_data = Pxx.transpose((0,2,1))

print(featured_data.shape)
# (13500, 51, 16)

Pxx = featured_data[13000,:,2]

# window_size = 3
# filtered_Pxx = np.convolve(Pxx, np.ones(window_size) / window_size, mode='same')

plt.semilogy(f, Pxx)
# plt.plot(f, Pxx)
plt.title(f'Canal C3 do Indivíduo 44')
plt.xlim([0, 50])
# plt.ylim([10e-18, 10e-12])
plt.xlabel('Frequência [Hz]')
plt.ylabel('PSD [$\mu$V²/Hz]')
plt.show()

with open(f'./datafiles/{dbname}/featuredData.pkl', 'wb') as file:
    pickle.dump(featured_data, file)
