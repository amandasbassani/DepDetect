import pickle
import numpy as np
import matplotlib.pyplot as plt

file1 = 'modmadata'
file2 = 'filtered_modmadata'

subject = 40
channel = 0

n = 75000  # numero de amostras
fs = 250

# Abrindo o arquivo pickle
with open(f'./datafiles/{file1}.pkl', 'rb') as f:
    data1 = pickle.load(f)

# Abrindo o arquivo pickle
with open(f'./datafiles/{file2}.pkl', 'rb') as f:
    data2 = pickle.load(f)

t = np.linspace(0, n/fs, n, False)  # 5 minutes (300 sec)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all')
ax1.plot(t, data1[subject,channel,:])
ax1.set_title('Original data')
ax2.plot(t, data2[subject,channel,:])
ax2.set_title('Filtered data')
ax2.set_xlabel('Time [seconds]')
plt.show()