import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

###
# DATA LOADING
###

data = keras.datasets.imdb

# Load data's num_words = 10000 => take 10k most frequent words.
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)


###
# DATA PRE-PROCESSING
###

# print(train_data[0])
word_index = data.get_word_index()
# +3 to account for 4 special chars
word_index = {k:(v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0     # padding; to make all reviews same length for processing, just to pad everything, which model will hopefully pick up on
word_index["<START>"] = 1   # start
word_index["<UNK>"] = 2     # unkown
word_index["<UNUSED>"] = 3  # unused

# I'ma be real with me. I'm lost and just following orders right about now. Hopefully going through the Stanford course later will lift the fog.
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# Want to pad everything so the set input/output neurons fit, though I guess a general intelligence wouldn't need it. But what am I, a god already? chill.
# So, either choose the longest review, or pick an arbitrarily length; cut off reviews longer, or pad ones shorter.
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=256)

def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

# for i in range(5):
#     print(decode_review(test_data[i]))


###
# MODEL
###

model = keras.Sequential()
model.add(keras.layers.Embedding(10000, 16))            # Embedding into 16D vectors for word closeness
model.add(keras.layers.GlobalAveragePooling1D())        # Average of the embedded vals
model.add(keras.layers.Dense(16, activation="relu"))    # find word patterns
model.add(keras.layers.Dense(1, activation="sigmoid"))  # Final output; good or bad?

model.summary()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

x_val = train_data[:10000]
y_val = train_labels[:10000]

x_train = train_data[10000:]
y_train = train_labels[10000:]

fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)
results = model.evaluate(test_data, test_labels)

print(results)

for i in range(5):
    test_review = test_data[i]
    predict = model.predict([test_review])
    print("Review #" + str(i) + ": ", decode_review(test_review))
    print("Prediction:", predict[i])
    print("Actual:", test_labels[i])
    print("\n\n")