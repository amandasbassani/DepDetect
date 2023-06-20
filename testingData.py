import tensorflow as tf
import json
import numpy as np
import pickle

with open('currentDB.json', 'r') as f:
    dbname = json.load(f)['dbname']

with open(f'./datafiles/{dbname}/info.json', 'r') as f:
    info = json.load(f)

trName = 'model_otim1'
classification = info['classification']

with open(f'./datafiles/{dbname}/models/test{trName}.pkl', 'rb') as f:
    test = pickle.load(f)

X_test = test[0]
y_test = test[1]

model = tf.keras.models.load_model(f'./datafiles/{dbname}/models/model{trName}.h5')

predict_x=model.predict(X_test)

y_pred = None
if classification == "sigmoid":
    y_pred = np.zeros(predict_x.shape)
    y_pred[predict_x>=0.5] = 1
elif classification == "softmax":
    y_pred = np.around(predict_x)

print("Test Accuracy:", np.mean(y_pred==y_test.astype(float)))
