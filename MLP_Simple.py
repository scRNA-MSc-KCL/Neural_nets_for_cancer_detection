import numpy as np
import pandas as pd
import sklearn
import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import requests
import zipfile
import csv
import scanpy as sc
import argparse
import anndata
import time
import os

#Load data
parser = argparse.ArgumentParser(description='Select dataset')
parser.add_argument('path', type = int)
start = time.time()

args = parser.parse_args()
#Dataset 1
if args.path == 1:
  labels =pd.read_csv("labels_1.csv", names = ["X"])
  data = sc.read("results_1.h5ad")
  file_loc = "DS1/MLP/Simple"
  b = 50
#Dataset 2
if args.path == 2:
  labels =pd.read_csv("labels_2.csv", names = ["X"])
  data = sc.read("results_2.h5ad")
  file_loc = "DS2/MLP/Simple"
  b = 500
#Dataset 3
if args.path == 4:
  labels =pd.read_csv("labels_4.csv", names = ["X"])
  data = sc.read("results_4.h5ad")
  file_loc = "DS4/MLP/Simple"
  b = 50

#Create output directory
path = os.getcwd()
path = os.path.join(path, "test_results/{}/{}".format(file_loc,start))
try:
  os.makedirs(path)
except OSError:
  print("Creation of the directory %s failed" % path)
else:
  print("Successfully created the directory %s" % path)

#define variables
num_lab = len(labels["X"].unique())
counter = 0
n = 500
a = "tanh"
o = "Adam"
e = 100
l = "categorical_crossentropy"

#create training and test sets
X_train, X_test, y_train, y_test = train_test_split(data.X, labels, test_size=0.2, random_state=42)
X_test, X_val, y_test, y_val= train_test_split(X_test, y_test, test_size=0.5, random_state=42)

#make labels for neural network catagorical
y_train = to_categorical(y_train, num_lab)
y_test = to_categorical(y_test, num_lab)
y_val = to_categorical(y_val, num_lab)

#Degine model
net = Sequential()
net.add(Dense(n, activation = a, input_shape = (data.n_vars,)))
net.add(Dense(num_lab, activation='softmax'))
net.compile(loss=l, optimizer=o)

#Train model
history = net.fit(X_train, y_train,validation_data=(X_val, y_val),epochs=e,batch_size=b)

#Test model
outputs = net.predict(X_test)
labels_predicted= np.argmax(outputs, axis=1)
y_test_decoded = np.argmax(y_test, axis=1)  # maybe change so you're not doing every time
correctly_classified =  (np.sum(labels_predicted == y_test_decoded)/(len(y_test_decoded)))*100
print("model number", counter)
print("accuracy", correctly_classified)

#Create plot figure
fig = plt.figure()
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
net.summary()

#Output results
from contextlib import redirect_stdout
with open('test_results/{}/{}/model_summary.txt'.format(file_loc, start), 'w') as fr:
  with redirect_stdout(fr):
    net.summary()
fig.savefig('test_results/{}/{}/fig_{}'.format(file_loc, start, counter))

end = time.time()
print("The time taken to complete this program was {}".format(end - start))
