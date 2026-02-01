import os
import pandas as pd
import numpy as np
import urllib.request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

file_path = "creditcard.csv"

if not os.path.exists(file_path):
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv",
        file_path
    )

data = pd.read_csv(file_path)

data["Txn_Count"] = data.index
data["Mean_Amount"] = data["Amount"].expanding().mean()
data["Time_Scaled"] = (data["Time"] - data["Time"].min()) / (data["Time"].max() - data["Time"].min())
data["Amount_Ratio"] = data["Amount"] / (data["Amount"].mean() + 1e-5)

X = data.drop("Class", axis=1)
y = data["Class"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.3,
    random_state=10,
    stratify=y
)

model = Sequential()
model.add(Dense(64, activation="relu", input_dim=X_train.shape[1]))
model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train,
    y_train,
    epochs=4,
    batch_size=64,
    validation_split=0.2
)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Test Accuracy:", accuracy)

sample = X_test[1].reshape(1, -1)
prob = model.predict(sample)[0][0]
print("Prediction Probability:", prob)

if prob > 0.5:
    print("Fraud Transaction")
else:
    print("Normal Transaction")
