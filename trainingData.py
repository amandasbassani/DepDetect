from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, GRU, Dense, Dropout
import pickle
import numpy as np
from numpy.random import seed
import random as python_random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

seed(1)
python_random.seed(1)

file = 'featured_data'

# Abrindo o arquivo pickle
with open(f'./datafiles/{file}.pkl', 'rb') as f:
    data = pickle.load(f)

# plt.plot(data[0,0,:])
# plt.show()

X = data[:,:-1,:].reshape((data.shape[0],(data.shape[1]-1)*data.shape[2]))

y = data[:,-1,0]
print(data.shape)


# Cria o modelo
model = Sequential()

model.add(Conv1D(filters=128, kernel_size=5, activation='relu', input_shape=(X.shape[1],1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=256, kernel_size=5, activation='relu', input_shape=(X.shape[1],1)))
model.add(GRU(units=256))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='softmax'))

# Compila o modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Divide os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

print(X_train.shape)
print(y_train.shape)
# Treina o modelo
history = model.fit(X_train, y_train, epochs=100, batch_size=256, validation_split=0.1)

# Avalia o modelo
loss, accuracy = history.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')
