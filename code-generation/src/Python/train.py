# Put at the beginning of your file:
####################################################################
# Following code is need to allow multiple people to share a GPU.
# Code limits TensorFlow's use of GPU memory to 0.2 of the maximum.
# Please don't be greedy with GPU memory until your final production
# run.
import tensorflow as tf
from keras import backend as K  # needed for mixing TensorFlow and Keras commands
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = .9
sess = tf.Session(config=config)
K.set_session(sess)
####################################################################

import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Lambda
from keras.layers import LSTM, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
# import tensorflow as tf
import math
from keras.models import load_model
import sys

infile, outfile = sys.argv[1], sys.argv[2]


print('loading data from', infile)
df = pd.read_csv(infile)
df = df.sample(frac=1).reset_index(drop=True)
df = df[df.output != 'E']
df.output = pd.to_numeric(df.output)

# df = df.head(50000)
total_progs = df.shape[0]
train_n = math.ceil(df.shape[0] * 0.8)
test_n = total_progs - train_n
train_x = df.iloc[:train_n,0:1].values[:,0]
train_y = df.iloc[:train_n,1].values.reshape((train_n, 1))
test_x = df.iloc[train_n:,0:1].values[:,0]
test_y = df.iloc[train_n:,1].values.reshape((test_n, 1))

print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)


SYMBOLS = ['(', ')', ' ', 'x', 'y', '1', '2', '0', 'λ', '+', '-', '*', '/']
N_SYMBOLS = len(SYMBOLS)
TOKENS = dict((c, i) for i, c in enumerate(SYMBOLS))
MAX_LEN = max(list(map(len, (list(train_x)+list(test_x)))))
print('max len:', MAX_LEN)

def tokenize_string(s):
    ret = np.zeros((MAX_LEN, N_SYMBOLS), dtype=bool)
    for i, char in enumerate(s):
        ret[i, TOKENS[char]] = 1
    return ret

def tokenize(a):
    return np.array(list(map(lambda s: tokenize_string(s), list(a))))



print('start tokenizing')
train_x = tokenize(train_x)
test_x = tokenize(test_x)

print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

model = Sequential()
# model.add(LSTM(512, return_sequences=True, input_shape=(MAX_LEN,N_SYMBOLS)))
# model.add(LSTM(512))
model.add(LSTM(256, input_shape=(MAX_LEN,N_SYMBOLS)))
model.add(Dropout(0.2))
# model.add(Dense(256))
# model.add(Activation('relu'))
# model.add(Dense(128))
# model.add(Activation('relu'))
model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['accuracy'])
model.fit(train_x,
          train_y,
          batch_size=800,
          epochs=200,
          validation_split=0.3,
          callbacks=[EarlyStopping(patience=6)])

print(test_y.std()/test_y.var())
print(model.evaluate(x=test_x, y=test_y, batch_size=800))
tests = tokenize(np.array(['(+ 2 1)',
                           '((λ) (x) (+ x 1)) 1)',
                           '(/ 1 2)',
                           '(* 2 (* 2 2))',
                           '(* 2 2)',
                           '(- 1 2)',
                           '(* 1 (+ 1 1))',
                           '(* 2 (+ 1 1))']))
print(model.predict(tests))
model.save(outfile)
