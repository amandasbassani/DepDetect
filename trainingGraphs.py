import matplotlib.pyplot as plt
import json
import pickle
import tensorflow as tf

trName = 'model_plothistory'
epochs = 100

with open('currentDB.json', 'r') as f:
    dbname = json.load(f)['dbname']

with open(f'./datafiles/{dbname}/info.json', 'r') as f:
    info = json.load(f)

# model = tf.keras.models.load_model(f'./datafiles/{dbname}/models/model{trName}.h5')

with open(f'./datafiles/{dbname}/models/history{trName}.pkl', 'rb') as f:
    history = pickle.load(f)

# Plotting both the curves simultaneously
plt.plot(range(epochs), history.history['loss'], color='r', label='Treinamento')
plt.plot(range(epochs), history.history['val_loss'], color='b', label='Validação')

# Naming the x-axis, y-axis and the whole graph
plt.xlabel("Época")
plt.ylabel("Perda")
plt.title("Gráfico de Perda")

# Adding legend, which helps us recognize the curve according to it's color
plt.legend()

# To load the display window
plt.show()

# Plotting both the curves simultaneously
plt.plot(range(epochs), history.history['Accuracy'], color='r', label='Treinamento')
plt.plot(range(epochs), history.history['val_Accuracy'], color='b', label='Validação')

# Naming the x-axis, y-axis and the whole graph
plt.xlabel("Época")
plt.ylabel("Acurácia")
plt.title("Gráfico de Acurácia")

# Adding legend, which helps us recognize the curve according to it's color
plt.legend()

# To load the display window
plt.show()