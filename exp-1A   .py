import numpy as np

inputs = np.array([1, 2, 2])

weights1 = np.array([
    [3, 4],
    [5, 6],
    [7, 8]
])

bias1 = np.array([1, 1])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

z1 = np.dot(inputs, weights1) + bias1
print("Hidden layer input:", z1)

a1 = sigmoid(z1)
print("Hidden layer output:", a1)

weights2 = np.array([
    [0.15],
    [0.25]
])

bias2 = np.array([0.8])

z2 = np.dot(a1, weights2) + bias2
final_output = sigmoid(z2)

print("Final output:", final_output)

y = 1
loss = (y - final_output) ** 2
print("Loss:", loss)

lr = 0.1

dloss_dout = -2 * (y - final_output)
dout_dz2 = final_output * (1 - final_output)
dz2_dw2 = a1.reshape(2, 1)

grad_w2 = dloss_dout * dout_dz2 * dz2_dw2

weights2 = weights2 - lr * grad_w2

print("Updated weights2:")
print(weights2)

