import pandas as pd
import tensorflow as tf
import numpy as np

## data of comments from naver shopping
raw = pd.read_table("C:\\Users\\dusjc\\Desktop\\CS\\codingapple\\tensorFlow\\Project 3\comments\\naver_shopping.txt", names=['rating', 'review'])
raw['label'] = np.where(raw['rating'] > 3, 1, 0) ## creating a new column on whether the review is nice or not

## preprocessing data (getting rid of special characters and english)
raw['review'] = raw['review'].str.replace(r"[^ㄱ-ㅎㅏ-ㅣ가-힣0-9 ]", "", regex=True)
raw.drop_duplicates(subset=['review'], inplace=True)

## bag of words
unique_words = raw['review'].tolist()
unique_words = ''.join(unique_words)
unique_words = list(set(unique_words))
unique_words.sort()

## preprocessing words using a tokenizer
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(char_level=True, oov_token="<OOV>") ## each character becomes a token
                                                          ## new characters become "does not exist"
word_list = raw['review'].tolist()
tokenizer.fit_on_texts(word_list)
train_seq = tokenizer.texts_to_sequences(word_list) ## converts all these reviews into numbers (tokenized)

## since all these reviews have different lengths we have to fill blank spaces with 0's
from keras.preprocessing.sequence import pad_sequences
X = pad_sequences(train_seq, maxlen=100) ## only count the first 100 characters
Y = raw['label'].tolist()

## Splitting the dataset into train/val/test
from sklearn.model_selection import train_test_split
trainX, valX, trainY, valY = train_test_split(X, Y, test_size=0.2, random_state=42) ## 20% of the data will be for validation
trainX = tf.convert_to_tensor(trainX)
valX = tf.convert_to_tensor(valX)
trainY = tf.convert_to_tensor(trainY)
valY = tf.convert_to_tensor(valY)

## model
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(5000, 16), ## used instead of one-hot-encoding
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.GRU(32, activation='tanh'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

## fitting
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(trainX, trainY, validation_data=(valX, valY), batch_size=64, epochs=5)