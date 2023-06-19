import pickle
import numpy as np
import matplotlib.pyplot as plt
import json

subject = 49
ch_name = 'T4'
filename1 = 'pickledData'
filename2 = 'filteredData'

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
tRec = info['tRec']
fs = info['fs']

n = fs*tRec
n_channels = len(ch_names)
t = np.linspace(0, n / fs, n, False)

if filename1 == 'pickledData':
    title1 = 'Dados Originais'
elif filename1 == 'filteredData':
    title1 = 'Dados Filtrados'
elif filename1 == 'cleanedData':
    title1 = 'Dados Limpos'

if filename2 == 'pickledData':
    title2 = 'Dados Originais'
elif filename2 == 'filteredData':
    title2 = 'Dados Filtrados'
elif filename2 == 'cleanedData':
    title2 = 'Dados Limpos'

fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all')
fig.suptitle(f'Canal {ch_name} do Indivíduo {subject + 1}')
ax1.plot(t, data1[subject,ch_names.index(ch_name),:], linewidth=1)
ax1.set_title(title1)
ax2.plot(t, data2[subject,ch_names.index(ch_name),:], linewidth=1)
ax2.set_title(title2)
ax2.set_xlabel('Time [s]')
plt.show()

dados_frequencia1 = np.fft.fft(data1[subject,ch_names.index(ch_name),:])
frequencias1 = np.fft.fftfreq(len(data1[subject,ch_names.index(ch_name),:]))
dados_frequencia2 = np.fft.fft(data2[subject,ch_names.index(ch_name),:])
frequencias2 = np.fft.fftfreq(len(data2[subject,ch_names.index(ch_name),:]))

fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all')
fig.suptitle(f'Canal {ch_name} do Indivíduo {subject + 1}')
ax1.plot(frequencias1*fs, np.abs(dados_frequencia1), linewidth=1)
ax1.set_title(title1)
ax2.plot(frequencias2*fs, np.abs(dados_frequencia2), linewidth=1)
ax2.set_title(title2)
ax2.set_xlabel('Frequência [Hz]')
plt.show()
