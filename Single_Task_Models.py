#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 21:07:14 2019

@author: geoff
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional 
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

# Read in the data
olid_training = pd.read_csv('./dataset/olid-training-processed.tsv', sep = '\t')
olid_test_A = pd.read_csv('./dataset/testset-levela-processed.tsv', sep = '\t')
olid_test_B = pd.read_csv('./dataset/testset-levelb-processed.tsv', sep = '\t')
olid_labels_A = pd.read_csv('./dataset/labels-levela.csv', header = None); olid_labels_A.columns = ['id', 'subtask_a']
olid_labels_B = pd.read_csv('./dataset/labels-levelb.csv', header = None); olid_labels_B.columns = ['id', 'subtask_b']

# Getting BERT ready
from bert_serving.client import BertClient
bc = BertClient()

print('Start training...')

# Get the data ready
X_train = bc.encode(olid_training['tweet'].tolist())            # BERT encode
X_train = X_train.reshape(13240, 1, 768)                        # Reshape for feeding to Bi-LSTM

y_train_a = olid_training['subtask_a']                          # Get Labels
y_train_a = np.where(y_train_a == "OFF", 1, 0)                  # Transform to Binary Labels

X_valid = bc.encode(olid_test_A['tweet'].tolist())
X_valid = X_valid.reshape(860, 1, 768)

y_valid_a = olid_labels_A['subtask_a']
y_valid_a = np.where(y_valid_a == "OFF", 1, 0)

# Create the Model
model_task_A = Sequential()
model_task_A.add(Bidirectional(LSTM(100, return_sequences = True), input_shape = (None, 768)))
model_task_A.add(Dropout(.2))
model_task_A.add(Bidirectional(LSTM(100)))
model_task_A.add(Dropout(.2))
model_task_A.add(Dense(32, activation = 'relu'))
model_task_A.add(Dropout(.2))
model_task_A.add(Dense(16, activation = 'relu'))
model_task_A.add(Dropout(.2))
model_task_A.add(Dense(4, activation = 'relu'))
model_task_A.add(Dropout(.2))
model_task_A.add(Dense(1, activation = 'sigmoid'))

model_task_A.compile(loss = 'binary_crossentropy', optimizer = "adam", metrics = ['accuracy'])
model_task_A.summary()

model_task_A.fit(X_train, y_train_a, epochs = 5, batch_size = 10)#, validation_data = [X_valid, y_valid_a])

y_pred_a = model_task_A.predict_classes(X_valid)
print(classification_report(y_pred_a, y_valid_a))

print('Task A finished.')

"""
Approximate Accuracy after 5 epochs = 79%
"""

# Get the data ready
filter = (olid_training['subtask_b'] == "UNT") | (olid_training['subtask_b'] == "TIN")
task_B_data = olid_training[filter]

X_train = bc.encode(task_B_data['tweet'].tolist())              # BERT encode
X_train = X_train.reshape(4400, 1, 768)                         # Reshape for feeding to Bi-LSTM

y_train_b = task_B_data['subtask_b']                            # Get Labels
y_train_b = np.where(y_train_b == "UNT", 1, 0)                  # Transform to Binary Labels

X_valid = bc.encode(olid_test_B['tweet'].tolist())
X_valid = X_valid.reshape(240, 1, 768)

y_valid_b = olid_labels_B['subtask_b']
y_valid_b = np.where(y_valid_b == "UNT", 1, 0)

# Create the Model
model_task_B = Sequential()
model_task_B.add(Bidirectional(LSTM(100, return_sequences = True), input_shape = (None, 768)))
model_task_B.add(Dropout(.2))
model_task_B.add(Bidirectional(LSTM(100)))
model_task_B.add(Dropout(.2))
model_task_B.add(Dense(32, activation = 'relu'))
model_task_B.add(Dropout(.2))
model_task_B.add(Dense(16, activation = 'relu'))
model_task_B.add(Dropout(.2))
model_task_B.add(Dense(4, activation = 'relu'))
model_task_B.add(Dropout(.2))
model_task_B.add(Dense(1, activation = 'sigmoid'))

model_task_B.compile(loss = 'binary_crossentropy', optimizer = "adam", metrics = ['accuracy'])
model_task_B.summary()

model_task_B.fit(X_train, y_train_b, epochs = 5, batch_size = 10)#, validation_data = [X_valid, y_valid_b])

y_pred_b = model_task_B.predict_classes(X_valid)
print(classification_report(y_pred_b, y_valid_b))
"""
Approximate Accuracy after 5 epochs = 100%
"""

print('Task B finished.')
