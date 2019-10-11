import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Naturally these should be sequestered to functions instead of this awful commenting out once no longer needed, but eh.

###
# SETTINGS
# ###
vocab_size = 88000
longest_review = 512


###
# DATA LOADING
###

data = keras.datasets.imdb

# Load data's num_words = 10000 => take 10k most frequent words.
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=vocab_size)


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

# # I'ma be real with me. I'm lost and just following orders right about now. Hopefully going through the Stanford course later will lift the fog.
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# # Want to pad everything so the set input/output neurons fit, though I guess a general intelligence wouldn't need it. But what am I, a god already? chill.
# # So, either choose the longest review, or pick an arbitrarily length; cut off reviews longer, or pad ones shorter.
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=longest_review)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=longest_review)

def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

for i in range(5):
    print(decode_review(test_data[i]))


##
# MODEL
##

model = keras.Sequential()
model.add(keras.layers.Embedding(10000, 16))            # Embedding into 16D vectors for word closeness
model.add(keras.layers.GlobalAveragePooling1D())        # Average of the embedded vals
model.add(keras.layers.Dense(16, activation="relu"))    # find word patterns
model.add(keras.layers.Dense(1, activation="sigmoid"))  # Final output; good or bad?

model.summary()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

x_val = train_data[:88000]
y_val = train_labels[:88000]

x_train = train_data[88000:]
y_train = train_labels[88000:]

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

def review_encode(s):
    encoded = [1] # <START> = 1
    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2) # = <UNK>
    return encoded

# model.save("text_classification.h5")
model = keras.models.load_model("text_classification.h5")
# Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2  
# This means we support AVX, which can speed up processing time by a shitload.
# If using GPU, ignore it; expensive operations dispatched to GPU and non-issue,
# Use import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'; to quell warning.
# Else: build tensorflow from the source optimized for particular CPU; ie not pip install

# Load in files and do some pre-processign
with open("example_review.txt", encoding="utf-8") as f:
    for line in f.readlines():
        nline = line.replace(",", "").replace(".", "").replace(":", "").replace(";", "").replace(")", "").replace("(", "").replace("\"", "").strip().split(" ")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=longest_review)
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict[0])
        
