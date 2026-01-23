import numpy as np

print("\n===== NUMPY: MANUAL BACKPROPAGATION =====")

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Input data (fraud = 1, normal = 0)
X = np.array([
    [0.2, 0.4, 0.6],
    [0.9, 0.1, 0.3],
    [0.1, 0.8, 0.5],
    [0.7, 0.2, 0.9]
])

y = np.array([[0], [1], [0], [1]])

# Initialize weights
np.random.seed(1)
W1 = np.random.rand(3, 4)
b1 = np.zeros((1, 4))

W2 = np.random.rand(4, 1)
b2 = np.zeros((1, 1))

lr = 0.1

# Training loop
for epoch in range(1000):

    # Forward pass
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, W2) + b2
    y_pred = sigmoid(z2)

    # Loss (Mean Squared Error)
    loss = np.mean((y - y_pred) ** 2)

    # Backpropagation
    d_output = (y - y_pred) * sigmoid_derivative(y_pred)
    d_hidden = d_output.dot(W2.T) * sigmoid_derivative(a1)

    # Weight updates
    W2 += a1.T.dot(d_output) * lr
    b2 += np.sum(d_output, axis=0, keepdims=True) * lr

    W1 += X.T.dot(d_hidden) * lr
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * lr

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

print("\nFinal NumPy Predictions:")
print(y_pred.round(3))

# -----------------------------
# PART 2: TENSORFLOW (AUTOMATIC BACKPROP)
# -----------------------------

print("\n===== TENSORFLOW: AUTOMATIC BACKPROPAGATION =====")

import tensorflow as tf

# Convert data to float32
X_tf = X.astype("float32")
y_tf = y.astype("float32")

# Build Feed Forward Neural Network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='sigmoid', input_shape=(3,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
    loss='mse'
)

# Train model (TensorFlow handles backprop automatically)
model.fit(X_tf, y_tf, epochs=1000, verbose=0)

# Predictions
print("\nFinal TensorFlow Predictions:")
print(model.predict(X_tf).round(3))
