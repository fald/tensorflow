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
train_data = keras.preprocessing.sequence.pad_sequence(train_data, value=word_index["<PAD>"], padding="post", maxLen=256)
test_data = keras.preprocessing.sequence.pad_sequence(test_data, value=word_index["<PAD>"], padding="post", maxLen=256)

def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

# for i in range(5):
#     print(decode_review(test_data[i]))


###
# MODEL
###

