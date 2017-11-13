import pandas as pd
import numpy as np
import time
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Dropout
import tensorflow as tf

df = pd.read_csv('../data/scheme-progs.csv')
df = df.sample(frac=1).reset_index(drop=True)
df = df[df.output != 'E']
df.output = pd.to_numeric(df.output)

train_x = df.iloc[:30000,0:1].values[:,0]
train_y = df.iloc[:30000,1].values.reshape((30000, 1))
test_x = df.iloc[30000:,0:1].values[:,0]
test_y = df.iloc[30000:,1].values.reshape((7389, 1))


SYMBOLS = ['(', ')', ' ', 'x', 'y', '1', '2', '0', '-', 'λ', '+', '-', '*', '/']
N_SYMBOLS = len(SYMBOLS)
TOKENS = dict((c, i) for i, c in enumerate(SYMBOLS))
MAX_LEN = max(list(map(len, (list(train_x)+list(test_x)))))

def tokenize_string(s):
    ret = np.zeros((MAX_LEN, N_SYMBOLS), dtype=bool)
    for i, char in enumerate(s):
        ret[i, TOKENS[char]] = 1
    return ret

def tokenize(a):
    return np.array(list(map(lambda s: tokenize_string(s), list(a))))


def BuildNet(NODES):
    model = Sequential()
    k = 1
    for nodes in NODES:
        if(k ==  1):
            model.add(LSTM(nodes,input_shape=(MAX_LEN,N_SYMBOLS)))
            model.add(Dropout(0.2))
        else:
            model.add(Dense(nodes))
            model.add(Activation('relu'))
        k = k+1
    model.add(Dense(1))
    model.compile(loss='categorical_crossentropy',
        optimizer='Adam',
        metrics=['accuracy']) 
    return model

train_x = tokenize(train_x)
test_x = tokenize(test_x)

BS = 1000 # batch size
ES = 10     # number of gradient step increases to tolerate
            # before early stopping.
EPOCHS       = 100
TRIALS       = 25
MAX_LAYERS   = 5
BS_CHOICES   = [1,10,100,1000] 
NODE_CHOICES = [2,4,8,16,32,64,128,256]
ES_CHOICES   = [0,5,10]


batch_size     = []
early_stopping = []
layer_nodes    = []
val_error      = []
total_time     = []

for trial in np.arange(TRIALS)+1:
    bs = np.random.choice(BS_CHOICES)
    batch_size.append(bs)
    es = np.random.choice(ES_CHOICES)
    early_stopping.append(es)
    NODES = []
    num_layers = np.random.choice(np.arange(MAX_LAYERS)+1)
    for i in np.arange(num_layers)+1:
        if i == num_layers:
            NODES.append(10)
        else:
            NODES.append(np.random.choice(NODE_CHOICES))
    layer_nodes.append(NODES)
    model = BuildNet(NODES)
    model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['accuracy'])
    time_start = time.time()
    hist = model.fit(train_x,train_y,
            batch_size=bs,
            epochs=EPOCHS,
            validation_split=0.2,
            verbose=0,
            callbacks=[EarlyStopping(patience=es)])
    err = 1 - hist.history['val_acc'][-1]
    val_error.append(err)
    time_stop = time.time()
    time_elapsed = time_stop - time_start
    total_time.append(time_elapsed)
    print('trial: %d/%d' % (trial,TRIALS))
df = pd.DataFrame()
df['nodes'] = layer_nodes
df['batch size'] = batch_size
df['early stopping'] = early_stopping
df['time (sec)'] = total_time
df['error'] = val_error 
df = df.sort_values('error')
print(df.round(3))