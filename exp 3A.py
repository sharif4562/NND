 import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, models

file_path = "creditcard.csv"

if not os.path.exists(file_path):
    import urllib.request
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv",
        file_path
    )

data = pd.read_csv(file_path)

features = data.iloc[:, :-1]
labels = data.iloc[:, -1]

scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(
    features_scaled,
    labels,
    test_size=0.25,
    random_state=1
)

model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_dim=X_train.shape[1]))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train,
    y_train,
    epochs=6,
    batch_size=64,
    verbose=1
)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Test Accuracy:", accuracy)

sample = X_test[0].reshape(1, -1)
prediction = model.predict(sample)[0][0]

print("Prediction Probability:", prediction)

if prediction > 0.5:
    print("Fraud Transaction")
else:
    print("Normal Transaction")

data = pd.read_csv("creditcard.csv")

features = data.iloc[:, :-1]
labels = data.iloc[:, -1]

scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(
    features_scaled,
    labels,
    test_size=0.25,
    random_state=1
)

model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_dim=X_train.shape[1]))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train,
    y_train,
    epochs=6,
    batch_size=64,
    verbose=1
)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
