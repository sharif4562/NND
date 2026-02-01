import numpy as np
import tensorflow as tf
# Input and target data
X = np.array([[1.0, 2.0, 3.0]])
Y = np.array([[1.0]])
# Define the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(4, activation='relu', input_dim=3))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
    loss='binary_crossentropy'
)
# Train the model
model.fit(X, Y, epochs=200, verbose=0)
# Make prediction
output = model(X)
print("Predicted Output:", output.numpy())
