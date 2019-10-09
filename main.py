import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# no autocomplete?

data = keras.datasets.fashion_mnist
# From tensorflow's site
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

(train_images, train_labels), (test_images, test_labels) = data.load_data()

# Normalize; img will look the same, but data is hella shrunk
train_images = train_images / 255.0
test_images = test_images / 255.0

# Build the braaaaaain
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"), # Rectify linear unit; fast and works well for variety of stuff.
    keras.layers.Dense(10, activation="softmax") # softmax -> vals add to 1
])

# adam is std optimizer, no more details yet.
# ditto the rest. Wow, useless tutorial tbh
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Actually train it
model.fit(train_images, train_labels, epochs=5)

# Results are in
# test_loss, test_acc = model.evaluate(test_images, test_labels)

# print("Tested accuracy:\t", test_acc)
# print("Test loss:\t", test_loss)

# If prediction a single item, encapsulate it in a list, as this is the expected data type.
prediction = model.predict(test_images)

for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual:" + class_names[test_labels[i]])
    plt.title("Prediction " + class_names[np.argmax(prediction[i])])
    plt.show()
# print(class_names[np.argmax(prediction[0])])