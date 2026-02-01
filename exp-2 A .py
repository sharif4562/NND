# Import required libraries
from google.colab import files
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
# Load and normalize MNIST data
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0
# Define the neural network
model = tf.keras.Sequential()
model.add(tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)
# Train the model
model.fit(
    train_images,
    train_labels,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)
# Evaluate model
loss, accuracy = model.evaluate(test_images, test_labels)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
# Upload handwritten image
print("\nUpload a handwritten digit image")
uploaded = files.upload()
file_name = list(uploaded.keys())[0]
# Image preprocessing
img = Image.open(file_name).convert("L")
img = img.resize((28, 28))
img_array = np.array(img)
# Normalize and invert if needed
img_array = 255 - img_array
img_array = img_array / 255.0
img_array = img_array.reshape(1, 28, 28)
# Prediction
predictions = model.predict(img_array)
predicted_digit = np.argmax(predictions)
# Display result
plt.imshow(img_array.reshape(28,28), cmap='gray')
plt.title(f"Predicted Digit: {predicted_digit}")
plt.axis('off')
plt.show()

print("Predicted digit is:", predicted_digit)

