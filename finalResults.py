import tensorflow as tf
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

model_name = 'last_modma2'
dbname = 'modma'

with open(f'./datafiles/{dbname}/models/test{model_name}.pkl', 'rb') as f:
# with open(f'./datafiles/mumtaz/models/test{model_name}.pkl', 'rb') as f:
    test = pickle.load(f)
model = tf.keras.models.load_model(f'./datafiles/{dbname}/models/model{model_name}.h5')
# model = tf.keras.models.load_model(f'./datafiles/mumtaz/models/model{model_name}.h5')

X_test = test[0]
y_test = test[1]


predict_x=model.predict(X_test)

y_pred = np.around(predict_x)
y_test = y_test.astype(float)


print("Test Accuracy:", np.mean(y_pred==y_test))
print("Test Loss:", model.evaluate(X_test, y_test))

# Gerar a matriz de confusão
y_pred = y_pred[:,1].astype(int)
y_test = y_test[:,1].astype(int)

print(precision_recall_fscore_support(y_test,y_pred, average='binary'))

cm = confusion_matrix(y_test, y_pred)

# Definir rótulos das classes
class_names = ['Saudável', 'Depressão']

# Plotar a matriz de confusão
sns.heatmap(cm, annot=True, cmap='Purples', fmt='d', xticklabels=class_names, yticklabels=class_names)

# Adicionar rótulos dos eixos
plt.xlabel('Classe Prevista')
plt.ylabel('Classe Verdadeira')

# Exibir o gráfico
plt.show()
