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

parser = argparse.ArgumentParser(description='Select dataset')
parser.add_argument('path', type = int)
start = time.time()
e = 100

args = parser.parse_args()
if args.path == 1:
  labels =pd.read_csv("labels_1.csv", names = ["X"])
  data = sc.read("results_1.h5ad")
  file_loc = "DS1/RBF"
  b = 50
if args.path == 2:
  labels =pd.read_csv("labels_2.csv", names = ["X"])
  data = sc.read("results_2.h5ad")
  file_loc = "DS2/RBF"
  b = 500
if args.path == 3:
  labels =pd.read_csv("labels_3.csv", names = ["X"])
  data = sc.read("results_3.h5ad")
  file_loc = "DS3/RBF"
  b = 2000
if args.path == 4:
  labels =pd.read_csv("labels_4.csv", names = ["X"])
  data = sc.read("results_4.h5ad")
  file_loc = "DS4/RBF"
  b = 50
  
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


net = Sequential()
net.add(RBFLayer(num_lab,initializer=InitCentersKMeans(X_train),betas=.01,input_shape=(data.n_vars,)))
net.add(Dense(num_lab, activation='softmax'))
net.compile(loss="categorical_crossentropy", optimizer="adam")
history = net.fit(X_train, y_train,validation_data=(X_val, y_val),epochs=e,batch_size=b)
outputs = net.predict(X_test)
labels_predicted= np.argmax(outputs, axis=1)
y_test_decoded = np.argmax(y_test, axis=1) 
correctly_classified =  (np.sum(labels_predicted == y_test_decoded)/(len(y_test_decoded)))*100
print("accuracy", correctly_classified)
fig = plt.figure()
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
fig.savefig('test_results/{}/{}/fig'.format(file_loc, start))


net.summary()
from contextlib import redirect_stdout
with open('test_results/{}/{}/model_summary.txt'.format(file_loc, start), 'w') as f:
    with redirect_stdout(f):
        net.summary()
f = open('test_results/{}/{}/model_summary.txt'.format(file_loc, start), 'a')
f.write("percentage accuracy on test set is {}\n".format(correctly_classified))
f.write("number of epochs is {}".format(e))
print("percentage accuracy; ", correctly_classified)
 

end = time.time()
print("The time taken to complete this program was {}".format(end - start))
