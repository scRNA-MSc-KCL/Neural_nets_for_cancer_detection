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
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
import requests
import zipfile
import csv
import scanpy as sc
import argparse
import anndata
import time
import os

parser = argparse.ArgumentParser(description='Select dataset')
parser.add_argument('path', type = int)
start = time.time()

args = parser.parse_args()
if args.path == 1:
  labels =pd.read_csv("labels_1.csv", names = ["X"])
  data = sc.read("results_1.h5ad")
  file_loc = "DS1/MLP/Final"
  b = 50
if args.path == 2:
  labels =pd.read_csv("labels_2.csv", names = ["X"])
  data = sc.read("results_2.h5ad")
  file_loc = "DS2/MLP/Final"
  b = 500
if args.path == 3:
  labels =pd.read_csv("labels_3.csv", names = ["X"])
  data = sc.read("results_3.h5ad")
  file_loc = "DS3/MLP/Final"
  b = 2000
if args.path == 4:
  labels =pd.read_csv("labels_4.csv", names = ["X"])
  data = sc.read("results_4.h5ad")
  file_loc = "DS4/MLP/Final"
  b = 50
  
path = os.getcwd()
path = os.path.join(path, "test_results/{}/{}".format(file_loc,start))
try:
  os.makedirs(path)
except OSError:
  print("Creation of the directory %s failed" % path)
else:
  print("Successfully created the directory %s" % path)

num_lab = len(labels["X"].unique())
counter = 0

#create training and test sets
X_train, X_test, y_train, y_test = train_test_split(data.X, labels, test_size=0.2, random_state=42)
X_test, X_val, y_test, y_val= train_test_split(X_test, y_test, test_size=0.5, random_state=42)

#make labels for neural network catagorical
y_train = to_categorical(y_train, num_lab)
y_test = to_categorical(y_test, num_lab)
y_val = to_categorical(y_val, num_lab)


accuracy_list = []
for i in range(100):
  net = Sequential()
  net.add(Dense(750, activation = "relu", input_shape = (data.n_vars,)))
  net.add(Dense(750, activation='relu'))
  net.add(Dense(num_lab, activation='softmax'))
  net.compile(loss="categorical_crossentropy", optimizer="Adam")
  history = net.fit(X_train, y_train,validation_data=(X_val, y_val),epochs=7,batch_size=b)
  outputs = net.predict(X_test)
  labels_predicted= np.argmax(outputs, axis=1)
  y_test_decoded = np.argmax(y_test, axis=1)  # maybe change so you're not doing every time
  correctly_classified =  (np.sum(labels_predicted == y_test_decoded)/(len(y_test_decoded)))*100
  print("model number", counter)
  print("accuracy", correctly_classified)
  accuracy_list.append(correctly_classified)


df = pd.DataFrame(list(zip(accuracy_list)),
                          columns =['accuracy'])

df.to_csv("test_results/{}/{}.csv".format(file_loc, start))



