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
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from numpy import savetxt

accuracy_list = []
counter = 0

parser = argparse.ArgumentParser(description='Select dataset')
parser.add_argument('path', type = int)
start = time.time()

#Load Data
args = parser.parse_args()

#Dataset 1
if args.path == 1:
  labels =pd.read_csv("labels_1.csv", names = ["X"])
  data = sc.read("results_1.h5ad")
  file_loc = "DS1/MLP/Final"
  b = 50
#Dataset 2
if args.path == 2:
  labels =pd.read_csv("labels_2.csv", names = ["X"])
  data = sc.read("results_2.h5ad")
  file_loc = "DS2/MLP/Final"
  b = 500
#Dataset 3
if args.path == 4:
  labels =pd.read_csv("labels_4.csv", names = ["X"])
  data = sc.read("results_4.h5ad")
  file_loc = "DS4/MLP/Final"
  b = 50

#Create working directory
path = os.getcwd()
path = os.path.join(path, "test_results/{}/{}".format(file_loc,start))
try:
  os.makedirs(path)
except OSError:
  print("Creation of the directory %s failed" % path)
else:
  print("Successfully created the directory %s" % path)

num_lab = len(labels["X"].unique())

#Separate training and test set
X_split, X_test, y_split, y_test = train_test_split(data.X, labels, test_size=0.2)
y_split = y_split.reset_index(drop = True)
y_test = y_test.reset_index(drop = True)
y_test = to_categorical(y_test, num_lab)

#Split training data
kf = KFold(n_splits=5, shuffle = True)
for train_index, test_index in kf.split(X_split):
  X_train, X_val = X_split[train_index], X_split[test_index]
  y_train, y_val = y_split['X'][train_index], y_split['X'][test_index]
  y_train = to_categorical(y_train, num_lab)
  y_val = to_categorical(y_val, num_lab)
  
  #Create model
  net = Sequential()
  if args.path == 1:
    net.add(Dense(500, activation = "tanh"))
  if args.path == 2:
    net.add(Dense(750, activation = "relu", input_shape = (data.n_vars,)))
    net.add(Dense(750, activation = "relu"))
  if args.path == 4:
    net.add(Dense(1200, activation = "relu", kernel_initializer = "glorot_normal", kernel_regularizer="l1_l2", input_shape = (data.n_vars,)))
    net.add(Dense(1300, activation = "relu", kernel_initializer = "glorot_normal", kernel_regularizer="l1_l2"))
  net.add(Dense(num_lab, activation='softmax'))
  
  #Train model
  net.compile(loss="categorical_crossentropy", optimizer="Adam")
  history = net.fit(X_train, y_train, validation_data=(X_val, y_val),epochs=7,batch_size=b)
  
  #Test model
  outputs = net.predict(X_test)
  labels_predicted= np.argmax(outputs, axis=1)
  y_test_decoded = np.argmax(y_test, axis=1)  # maybe change so you're not doing every time
  correctly_classified =  (np.sum(labels_predicted == y_test_decoded)/(len(y_test_decoded)))*100
  print("model number", counter)
  print("accuracy", correctly_classified)
  with open('test_results/{}/{}/summary{}.txt'.format(file_loc, start, counter), 'w') as fr:
    fr.write("precision score: {}".format(precision_score(y_test_decoded, labels_predicted, average=None)))
    fr.write("recall score: {} ".format(recall_score(y_test_decoded, labels_predicted, average=None)))
    fr.write("accuracy: {}".format(correctly_classified))
  print(precision_score(y_test_decoded, labels_predicted, average=None))
  print(recall_score(y_test_decoded, labels_predicted, average=None))
  savetxt("test_results/{}/{}/{}_ypred.csv".format(file_loc, start, counter), labels_predicted, delimiter=',')
  savetxt("test_results/{}/{}/{}_ytrue.csv".format(file_loc, start, counter), y_test_decoded, delimiter=',')
  counter +=1
  accuracy_list.append(correctly_classified)

#output results
print(accuracy_list)
df = pd.DataFrame(list(zip(accuracy_list)),columns =['accuracy_list'])
df.to_csv("test_results/{}/{}.csv".format(file_loc, start))
end = time.time()
print("The time taken to complete this program was {}".format(end - start))
