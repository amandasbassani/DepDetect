from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, GRU, Dense, Dropout
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import json

np.random.seed(0)

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
[trn, val, tst] = info['trProp']

labels = labeldata * tRec
X = data
y = np.array(labels)

model = Sequential()

model.add(Conv1D(filters=128, kernel_size=5, activation='relu', input_shape=(X.shape[1],X.shape[2])))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=256, kernel_size=5, activation='relu', input_shape=(X.shape[1],X.shape[2])))
model.add(GRU(units=256))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='softmax'))

# Compila o modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Divide os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tst, random_state=42)

# Treina o modelo
history = model.fit(X_train, y_train, epochs=100, batch_size=256, validation_split=val/(val+trn))

# Avalia o modelo
loss, accuracy = history.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')
