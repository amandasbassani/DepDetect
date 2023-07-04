import pickle
import numpy as np
from scipy.signal import welch, periodogram
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

downsamp = 2
data = data[:,:,0:256:2]
print(data.shape)

f, Pxx = welch(x=data,
               fs=float(fs)/downsamp,
               window='hamming',
               nperseg=15,
               # noverlap=7,
               axis=2)

# f, Pxx = welch(x=data,
#                fs=float(fs),
#                window='hamming',
#                nperseg=100,
#                noverlap=50,
#                axis=2)

# f, Pxx = periodogram(x=data,
#                      fs=float(fs),
#                      window='hamming',
#                      axis=2)

featured_data = Pxx.transpose((0,2,1))

print(featured_data.shape)
# (13500, 51, 16)
print(f)

Pxx = featured_data[0,:,2]

plt.semilogy(f, Pxx)
# plt.plot(f, Pxx)
plt.title(f'Canal C3 do Indivíduo 44')
plt.xlim([0,60])
plt.xticks(ticks=range(0,60,5))
plt.ylim([10e-7, 10e0])
plt.xlabel('Frequência [Hz]')
plt.ylabel('PSD [$\mu$V²/Hz]')
plt.show()

# plt.semilogy(f, Pxx)
# # plt.plot(f, Pxx)
# plt.title(f'Canal C3 do Indivíduo 44')
# plt.xlim([0, 128])
# plt.xticks(ticks=range(0,129,8),labels=['' if i % 16 != 0 else i for i in range(0, 129, 8)])
# plt.ylim([10e-7, 10e0])
# plt.xlabel('Frequência [Hz]')
# plt.ylabel('PSD [$\mu$V²/Hz]')
# plt.show()


with open(f'./datafiles/{dbname}/featuredData.pkl', 'wb') as file:
    pickle.dump(featured_data, file)
