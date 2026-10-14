import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("HANDWRITTEN DIGIT RECOGNITION (Deep Learning)")
print("-" * 48)

# MNIST contains 28x28 grayscale images of handwritten digits 0-9.
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixel values from 0..255 to 0..1.
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

model = keras.Sequential([
    layers.Input(shape=(28, 28)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()
model.fit(x_train, y_train, epochs=3, batch_size=128, validation_split=0.1, verbose=1)

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest accuracy: {accuracy * 100:.2f}%")

# Predict one test image and save a sample output figure.
index = 0
probabilities = model.predict(x_test[index:index+1], verbose=0)[0]
predicted_digit = int(np.argmax(probabilities))
actual_digit = int(y_test[index])

print(f"Actual digit: {actual_digit}")
print(f"Predicted digit: {predicted_digit}")

plt.imshow(x_test[index], cmap="gray")
plt.title(f"Actual: {actual_digit} | Predicted: {predicted_digit}")
plt.axis("off")
plt.tight_layout()
plt.savefig("digit_prediction.png")
print("Sample prediction saved as digit_prediction.png")

model.save("digit_recognition_model.keras")
print("Trained model saved as digit_recognition_model.keras")
