import pickle
import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt

filename = 'filtered_modmadata'
fs = 250

# Abrindo o arquivo pickle
with open(f'./datafiles/{filename}.pkl', 'rb') as f:
    pkdata = pickle.load(f)

# Abrindo o arquivo pickle
with open(f'./datafiles/label_modmadata.pkl', 'rb') as f:
    labeldata = pickle.load(f)

def labeling_data(data, data_label):
    names_matrix = []

    for lab in data_label:
        name_matrix = np.zeros((1, 1, data.shape[2]))
        name_matrix[:] = lab
        names_matrix.append(name_matrix)

    names_matrix = np.vstack(names_matrix)

    return np.hstack((data, names_matrix))

totaldata = labeling_data(pkdata,labeldata)
print(f'pkdata.shape: {pkdata.shape}')
print(f'labeldata.shape: {labeldata.shape}')
print(f'totaldata.shape: {totaldata.shape}')
# Cutting every 1 s
time_cut = 1  # seconds

inicialdata = []
for i in range(0,totaldata.shape[2],time_cut*fs):
    inicialdata.append(totaldata[:,:,i:i+time_cut*fs])

inicialdata = np.vstack(inicialdata)
print(f'incialaldata.shape: {inicialdata.shape}')

f, Pxx = welch(x=inicialdata[:,:-1,:],
               fs=float(fs),
               window='hamming',
               nperseg=100,
               axis=2)


# Diminuindo o vetor de features
n = 21
Pxx = Pxx[:,:,:n]
f = f[0:n]

labels = inicialdata[:,-1,:].reshape((inicialdata[:,-1,:].shape[0],1,inicialdata[:,-1,:].shape[1]))[:,:,:n]

plt.plot(f,Pxx[13,0,:])
plt.show()

featured_data = np.hstack((Pxx,labels))

with open(f'./datafiles/featured_data.pkl', 'wb') as file:
    pickle.dump(featured_data, file)
