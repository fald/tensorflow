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

if __name__ == "__main__":
    plt.imshow(train_images[0], cmap=plt.cm.binary)
    plt.show()