import pandas as pd

# Load fraud dataset
df = pd.read_csv("creditcard.csv")

print(df.head())

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load data
df = pd.read_csv("creditcard.csv")
# 2. Split into X and y
X = df.drop("Class", axis=1)
y = df["Class"]

# 3. Scale numeric features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 5. Simple Feedforward NN
model = tf.keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# 6. Compile
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 7. Train
model.fit(X_train, y_train, epochs=10, batch_size=64)

# 8. Evaluate
loss, acc = model.evaluate(X_test, y_test)
print("Test accuracy:", acc)
