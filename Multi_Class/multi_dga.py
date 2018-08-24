from __future__ import print_function
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
from keras.utils.np_utils import to_categorical
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn import metrics
from sklearn.preprocessing import Normalizer
import h5py
from keras import callbacks
from keras.callbacks import CSVLogger
import keras
import keras.preprocessing.text
import itertools
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras import callbacks

#trainlabels = pd.read_csv('dgcorrect/trainlabel.csv', header=None)
#trainlabel = trainlabels.iloc[:,0:1]
#testlabels = pd.read_csv('dgcorrect/testlabel.csv', header=None)
#testlabel = testlabels.iloc[:,0:1]


#train = pd.read_csv('dgcorrect/train.txt', header=None)
#test = pd.read_csv('dgcorrect/test.txt', header=None)

train_data = pd.read_csv("train.csv", header=None).values
test_data1 = pd.read_csv("test1.txt", header=None).values
test_data2 = pd.read_csv("testing-2.txt", header=None).values

#train = train_data[:,0]
#trainlabel = train_data[:,1]

# Extract data and labels
X = [x[0] for x in train_data]
labels = [x[1] for x in train_data]

T1 = [x[0] for x in test_data1]
T2 = [x[0] for x in test_data2]
#labels_test = [x[1] for x in test_data]


# Generate a dictionary of valid characters
valid_chars = {x:idx+1 for idx, x in enumerate(set(''.join(X+T1+T2)))}

max_features = len(valid_chars) + 1
maxlen = np.max([len(x) for x in X])

# Convert characters to int and pad
X = [[valid_chars[y] for y in x] for x in X]
X_train = sequence.pad_sequences(X, maxlen=maxlen)

y_train1 = np.array(labels)
y_train = to_categorical(y_train1)

T1 = [[valid_chars[y] for y in x] for x in T1]
X_test1 = sequence.pad_sequences(T1, maxlen=maxlen)

T2 = [[valid_chars[y] for y in x] for x in T2]
X_test2 = sequence.pad_sequences(T2, maxlen=maxlen)


print(X_train.shape)
print(y_train.shape)


embedding_vecor_length = 128

model = Sequential()
model.add(Embedding(max_features, embedding_vecor_length, input_length=maxlen))
model.add(LSTM(128))
model.add(Dropout(0.1))
model.add(Dense(21))
model.add(Activation('softmax'))
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
checkpointer = callbacks.ModelCheckpoint(filepath="logs/lstm/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='val_acc',mode='max')
csv_logger = CSVLogger('logs/lstm/training_set_lstmanalysis.csv',separator=',', append=False)
model.fit(X_train, y_train, batch_size=32, nb_epoch=1000,validation_split=0.33, shuffle=True,callbacks=[checkpointer,csv_logger])
#score, acc = model.evaluate(X_test, y_test, batch_size=32)
#print('Test score:', score)
#print('Test accuracy:', acc)

y_pred1 = model.predict_classes(X_test1)
y_pred2 = model.predict_classes(X_test2)
#np.savetxt('multi_predict.csv', X_test, fmt='%i', delimiter=',')
np.savetxt('binary_predict1.csv', y_pred1, fmt='%i', delimiter=',')
np.savetxt('binary_predict2.csv', y_pred2, fmt='%i', delimiter=',')

'''
embedding_vecor_length = 128
model = Sequential()
model.add(Embedding(max_features, embedding_vecor_length, input_length=maxlen))
model.add(LSTM(128))
model.add(Dropout(0.1))
model.add(Dense(1))
model.add(Activation('sigmoid'))
print(model.summary())
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
checkpointer = callbacks.ModelCheckpoint(filepath="logs/lstm/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='val_acc',mode='max')
csv_logger = CSVLogger('logs/lstm/training_set_lstmanalysis.csv',separator=',', append=False)

model.fit(X_train, y_train, batch_size=32, nb_epoch=1,validation_split=0.33, shuffle=True,callbacks=[checkpointer,csv_logger])

score, acc = model.evaluate(X_test, y_test, batch_size=32)
print('Test score:', score)
print('Test accuracy:', acc)
'''
