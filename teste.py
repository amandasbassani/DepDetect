import numpy as np

def zscore_normalize_channels(X):
    n_samples, n_features, n_channels = X.shape

    # Reshape the input array for easy computation
    X_reshaped = X.reshape(n_samples * n_features, n_channels)

    # Calculate the mean and standard deviation for each channel
    channel_means = np.mean(X_reshaped, axis=0)
    channel_stds = np.std(X_reshaped, axis=0)

    # Subtract mean and divide by standard deviation for each channel
    normalized_X = (X - channel_means) / channel_stds

    return normalized_X

# Example usage
X = np.random.rand(100, 50, 3)  # Sample multi-channel data

# Apply z-score normalization to each channel
normalized_X = zscore_normalize_channels(X)

# Check the shape of the normalized array
print(X[0:3,0:3,0:3])
print(normalized_X[0:3,0:3,0:3])  # Output: (100, 50, 3)

print(sum([0.0344782, -1.53763609, -1.29792641]))

"""
# example making new class predictions for a classification problem
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
import numpy as np
# generate 2d classification dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
scalar = MinMaxScaler()
scalar.fit(X)
X = scalar.transform(X)
# define and fit the final model
model = Sequential()
model.add(Dense(4, input_shape=(2,), activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(X, y, epochs=500, verbose=0)
# new instances where we do not know the answer
Xnew, _ = make_blobs(n_samples=3, centers=2, n_features=2, random_state=1)
print(Xnew)
Xnew = scalar.transform(Xnew)
print(Xnew)
# make a prediction
# ynew = model.predict_classes(Xnew)
predict_x=model.predict(Xnew)
ynew=np.argmax(predict_x,axis=1)
# show the inputs and predicted outputs
for i in range(len(Xnew)):
 print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))
 
"""