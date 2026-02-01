from google.colab import files
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=3)

print("Upload your two-digit image (example: 45, 78, 23)")
uploaded = files.upload()
image_name = list(uploaded.keys())[0]

img = Image.open(image_name).convert("L")
img = ImageOps.invert(img)
img = img.resize((56, 28))

img_array = np.array(img) / 255.0
digit1 = img_array[:, :28].reshape(1, 28, 28)
digit2 = img_array[:, 28:].reshape(1, 28, 28)
pred1 = np.argmax(model.predict(digit1))
pred2 = np.argmax(model.predict(digit2))
two_digit_number = str(pred1) + str(pred2)
plt.imshow(img_array, cmap='gray')
plt.title(f"Predicted Two-Digit Number: {two_digit_number}")
plt.axis('off')
plt.show()
print("Predicted two-digit number is:", two_digit_number)

