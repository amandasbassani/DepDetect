import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import json
import os

with open('currentDB.json', 'r') as f:
    dbname = json.load(f)['dbname']

with open(f'./datafiles/{dbname}/info.json', 'r') as f:
    info = json.load(f)

with open(f'./datafiles/{dbname}/featuredData.pkl', 'rb') as f:
    data = pickle.load(f)

with open(f'./datafiles/{dbname}/dataLabels.pkl', 'rb') as f:
    labeldata = pickle.load(f)

pathmodels = f'./datafiles/{dbname}/models/'

if not os.path.exists(pathmodels):
    os.makedirs(pathmodels)

tf.keras.utils.set_random_seed(0)

fs = info['fs']
bands = info['bands']
window_time = info['window_time']//1000
tRec = info['tRec']
trn, val, tst = info['trProp']
trName = info['trName']

labels = labeldata * (tRec//window_time)
X = data
y = np.array(labels)
print(X.shape)
print(y.shape)


def zscore(x1, x2):
    u = np.mean(x1,axis=0)
    n = x1.shape[0]
    std = np.sqrt(np.sum((x1-u)**2,axis=0)/(n-1))
    return (x1-u)/std, (x2-u)/std


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tst, random_state=42)
X_train, X_test = zscore(X_train, X_test)

print(X_train.shape)
print(X_test.shape)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu', input_shape=(X.shape[1], X.shape[2])))
model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
model.add(tf.keras.layers.Conv1D(filters=256, kernel_size=5, activation='relu'))
model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
model.add(tf.keras.layers.GRU(units=256))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(units=1, activation='softmax'))

optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.005)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['Accuracy', 'Recall', 'Precision'])

model.fit(X_train, y_train, epochs=100, batch_size=256, validation_split=0.1)

with open(f'./datafiles/{dbname}/models/test{trName}.pkl', 'wb') as f:
    pickle.dump([X_test, y_test], f)

model.save(f'./datafiles/{dbname}/models/model{trName}.h5')
