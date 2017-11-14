import math
import sys
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Lambda
from keras.layers import LSTM, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import tensorflow as tf

infile, outfile = sys.argv[1], sys.argv[2]

df = pd.read_csv(infile)
df = df.sample(frac=1).reset_index(drop=True)
df = df.head(200000)
df['noerr'] = 1 - df['error']


total_progs = df.shape[0]
train_n = math.ceil(df.shape[0] * 0.8)
test_n = total_progs - train_n
train_x = df.iloc[:train_n,0:1].values[:,0]
train_y = df.iloc[:train_n,2:].values
test_x = df.iloc[train_n:,0:1].values[:,0]
test_y = df.iloc[train_n:,2:].values

print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

SYMBOLS = ['(', ')', ' ', 'x', 'y', '1', '2', '0', '-', 'λ', '+', '-', '*', '/']
N_SYMBOLS = len(SYMBOLS) 
TOKENS = dict((c, i) for i, c in enumerate(SYMBOLS))
MAX_LEN = max(list(map(len, (list(train_x)+list(test_x)))))

def tokenize_string(s):
    ret = np.zeros((MAX_LEN, N_SYMBOLS + 1), dtype=bool)
    for i, char in enumerate(s):
        ret[i, TOKENS[char]] = 1
    for i in range(len(s), MAX_LEN):
        ret[i, N_SYMBOLS] = 1
    return ret

print(tokenize_string('(+ 1 2)'))

def tokenize(a):
    return np.array(list(map(lambda s: tokenize_string(s), list(a))))

train_x = tokenize(train_x)
test_x = tokenize(test_x)

print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

model = Sequential()
model.add(LSTM(256, input_shape=(MAX_LEN,N_SYMBOLS+1)))
model.add(Dropout(0.2))
# model.add(Dense(128))
# model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.fit(train_x,
          train_y,
          batch_size=800,
          epochs=200,
          validation_split=0.3,
          callbacks=[EarlyStopping(patience=5)])

print(model.evaluate(x=test_x, y=test_y, batch_size=800))

model.save(outfile)

tests = tokenize(np.array(['(/ 1 1)', '((λ (x) x) 1)', '((λ (x) y) 1)', '(/ 2 0)', '(/ 0 1)', '(/ 1 (- 1 1))']))

print(model.predict(tests))
