from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, GRU, Dense, Dropout
import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
from keras.optimizers import SGD
opt = SGD(learning_rate=0.005)

tf.keras.utils.set_random_seed(0)

with open('currentDB.json', 'r') as f:
    dbname = json.load(f)['dbname']

with open(f'./datafiles/{dbname}/info.json', 'r') as f:
    info = json.load(f)

with open(f'./datafiles/{dbname}/featuredData.pkl', 'rb') as f:
    data = pickle.load(f)

with open(f'./datafiles/{dbname}/dataLabels.pkl', 'rb') as f:
    labeldata = pickle.load(f)

fs = info['fs']
bands = info['bands']
window_time = info['window_time']//1000
tRec = info['tRec']
trn, val, tst = info['trProp']
trName = info['trName']

labels = labeldata * (tRec//window_time)
X = data
y = np.array(labels)
print(y.shape)
print(X.shape)

def zscore(x1, x2):
    u = np.mean(x1,axis=0)
    n = x1.shape[0]
    std = np.sqrt(np.sum((x1-u)**2,axis=0)/(n-1))
    return (x1-u)/std, (x2-u)/std

batch_size = 256
input_shape = (batch_size, X.shape[1], X.shape[2])
print(input_shape)

# x = tf.random.normal(input_shape)
# y = tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu',input_shape=input_shape[1:])(x)
# print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tst, random_state=42)
X_train, X_test = zscore(X_train, X_test)

# X_train = X[:14310,:,:]
# X_test = X[14310:,:,:]
# y_train = y[:14310,:,:]
# y_test = y[14310:,:,:]

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu', input_shape=input_shape[1:]))
model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
model.add(tf.keras.layers.Conv1D(filters=256, kernel_size=5, activation='relu'))
model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
model.add(tf.keras.layers.GRU(units=256))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(units=1, activation='softmax'))

optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.005)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100, batch_size=batch_size, validation_split=0.1)

with open(f'./datafiles/{dbname}/models/test{trName}.pkl', 'wb') as f:
    pickle.dump([X_test, y_test], f)

model.save(f'./datafiles/{dbname}/models/model{trName}.h5')

# Evaluate the model
# loss, accuracy = model.evaluate(X_test, y_test)
# print("Test Loss:", loss)
# print("Test Accuracy:", accuracy)

