  
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
from rbflayer import RBFLayer, InitCentersRandom
from kmeans_initializer import InitCentersKMeans
from contextlib import redirect_stdout

parser = argparse.ArgumentParser(description='Select dataset')
parser.add_argument('path', type = int)
start = time.time()

args = parser.parse_args()
if args.path == 1:
  labels =pd.read_csv("labels_1.csv", names = ["X"])
  data = sc.read("results_1.h5ad")
  file_loc = "DS1/RBF/Final"
  b = 50
  #e = 200
  e = 1
  #number_of_models = 100
  number_of_models = 1
  be = 0.001
if args.path == 2:
  labels =pd.read_csv("labels_2.csv", names = ["X"])
  data = sc.read("results_2.h5ad")
  file_loc = "DS2/RBF/Final"
  b = 500
  e = 100
  number_of_models = 10
  be = 0.001
if args.path == 4:
  labels =pd.read_csv("labels_4.csv", names = ["X"])
  data = sc.read("results_4.h5ad")
  file_loc = "DS4/RBF/Final"
  b = 50
  e = 600
  number_of_models = 100
  be = 0.0001
  
path = os.getcwd()
path = os.path.join(path, "test_results/{}/{}".format(file_loc,start))
try:
  os.makedirs(path)
except OSError:
  print("Creation of the directory %s failed" % path)
else:
  print("Successfully created the directory %s" % path)

X_train, X_test, y_train, y_test = train_test_split(data.X, labels, test_size=0.2, random_state=42)
X_test, X_val, y_test, y_val= train_test_split(X_test, y_test, test_size=0.5, random_state=42)

#number of labels
num_lab = len(labels["X"].unique())

#make labels for neural network catagorical
y_train = to_categorical(y_train, num_lab)
y_test = to_categorical(y_test, num_lab)
y_val = to_categorical(y_val, num_lab)

accuracy_list = []

for i in range(number_of_models):
  net = Sequential()
  net.add(RBFLayer(num_lab,initializer=InitCentersKMeans(X_train),betas=be,input_shape=(data.n_vars,)))
  net.add(Dense(num_lab, activation='softmax'))
  net.compile(loss="categorical_crossentropy", optimizer="Adamax")
  history = net.fit(X_train, y_train,validation_data=(X_val, y_val),epochs=e,batch_size=b)
  outputs = net.predict(X_test)
  labels_predicted= np.argmax(outputs, axis=1)
  y_test_decoded = np.argmax(y_test, axis=1) 
  correctly_classified =  (np.sum(labels_predicted == y_test_decoded)/(len(y_test_decoded)))*100
  print("accuracy", correctly_classified)
  net.summary()
  f = open('test_results/{}/{}/model_summary.txt'.format(file_loc, start), 'a')
  f.write("percentage accuracy on test set is {}\n".format(correctly_classified))
  f.write("number of epochs is {}".format(e))
  print("percentage accuracy; ", correctly_classified)
  accuracy_list.append(correctly_classified)
  with open('test_results/{}/{}/model_summary.txt'.format(file_loc, start), 'w') as f:
    with redirect_stdout(f):
      net.summary()

  
df = pd.DataFrame(list(zip(accuracy_list)),
                          columns =['accuracy'])
end = time.time()
print("The time taken to complete this program was {}".format(end - start))
df.to_csv("test_results/{}/{}.csv".format(file_loc, start))
