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

parser = argparse.ArgumentParser(description='Select dataset')
parser.add_argument('path', type = int)
start = time.time()

args = parser.parse_args()
if args.path == 1:
  labels =pd.read_csv("labels_1.csv", names = ["X"])
  data = sc.read("results_1.h5ad")
  file_loc = "DS1/MLP"
  b = 50
if args.path == 2:
  labels =pd.read_csv("labels_2.csv", names = ["X"])
  data = sc.read("results_2.h5ad")
  file_loc = "DS2/MLP"
  b = 500
if args.path == 3:
  labels =pd.read_csv("labels_3.csv", names = ["X"])
  data = sc.read("results_3.h5ad")
  file_loc = "DS3/MLP"
  b = 2000
if args.path == 4:
  labels =pd.read_csv("labels_4.csv", names = ["X"])
  data = sc.read("results_4.h5ad")
  file_loc = "DS4/MLP"
  b = 50
  
path = os.getcwd()
path = os.path.join(path, "test_results/{}/{}".format(file_loc,start))
try:
  os.makedirs(path)
except OSError:
  print("Creation of the directory %s failed" % path)
else:
  print("Successfully created the directory %s" % path)

num_lab = len(labels)
counter = 0

#create training and test sets
X_train, X_test, y_train, y_test = train_test_split(data.X, labels, test_size=0.2, random_state=42)
X_test, X_val, y_test, y_val= train_test_split(X_test, y_test, test_size=0.5, random_state=42)

#make labels for neural network catagorical
y_train = to_categorical(y_train, num_lab)
y_test = to_categorical(y_test, num_lab)
y_val = to_categorical(y_val, num_lab)


#Neural network testing function
def MLP_Assembly(optimizer, loss_function, X_train, y_train, X_test, y_test, X_val, y_val, epoch, Nodes, activation, counter, num_lab, b, layer_number):
  optimizer_list = []
  loss_function_list = []
  epoch_list = []
  node_1_length_list = []
  activation_layer_1_list = []
  accuracy_list = []
  layer_number_list = []
  for o in optimizer:
    for l in loss_function:
      for e in epoch:
        for n1 in Nodes:
          for a1 in activation:
            for lr in layer_number:
              net = Sequential()
              if lr == 1:
                net.add(Dense(n1, activation = a1, input_shape = (data.n_vars,)))
              if lr == 2:
                net.add(Dense(n1, activation = a1, input_shape = (data.n_vars,)))
                net.add(Dense(n1, activation = a1))
              if lr == 3:
                net.add(Dense(n1, activation = a1, input_shape = (data.n_vars,)))
                net.add(Dense(n1, activation = a1))
                net.add(Dense(n1, activation = a1))
              net.add(Dense(num_lab, activation='softmax'))
              counter += 1
              layer_number_list.append(lr)
              optimizer_list.append(o)
              loss_function_list.append(l)
              activation_layer_1_list.append(a1)
              epoch_list.append(e)
              node_1_length_list.append(n1)
              net.compile(loss=l, optimizer=o)
              history = net.fit(X_train, y_train,
                                validation_data=(X_val, y_val),
                                epochs=e,batch_size=b)
              outputs = net.predict(X_test)
              labels_predicted= np.argmax(outputs, axis=1)
              y_test_decoded = np.argmax(y_test, axis=1)  # maybe change so you're not doing every time
              correctly_classified =  (np.sum(labels_predicted == y_test_decoded)/(len(y_test_decoded)))*100
              accuracy_list.append(correctly_classified)
              print("model number", counter)
              print("accuracy", correctly_classified)
              fig = plt.figure()
              plt.plot(history.history['loss'], label='training loss')
              plt.plot(history.history['val_loss'], label='validation loss')
              plt.xlabel('epochs')
              plt.ylabel('loss')
              plt.legend()
              fig.savefig('test_results/{}/{}/fig_{}'.format(file_loc, start, counter))
  print(accuracy_list)
  df = pd.DataFrame(list(zip(optimizer_list, loss_function_list, epoch_list, node_1_length_list, activation_layer_1_list, accuracy_list, layer_number_list)),
                        columns =['optimizer', 'loss_function', "epochs", "node1_length", "activation_layer1", "perceptage accuracy", "layer_number"])
  return df

#define variables
#Nodes = np.arange(50, 2050, 500)
Nodes = [1200]*100
#Nodes = np.arange(10, 3010, 10)
#activation = ["tanh", "relu", "sigmoid", "softplus", "softsign", "selu", "elu"]
activation = ["relu"]
#optimizer = ["SGD", "RMSprop", "Adam", "Adadelta", "Adagrad", "Adamax", "Nadam", "Ftrl"]
optimizer = ["Adamax"]
#epoch = [100]
epoch = [7]
layer_number = [1]

#loss_function = ["categorical_crossentropy", "poisson","kl_divergence"]
loss_function = ["categorical_crossentropy"]
#consider using custom learning rate
#may or may not get used. See impact on above results
#regularizer = ["l1", "l2", "l1_l2"]
#kernal_init = ["random_normal", "random_uniform", "truncated_normal", "zeros", "ones", "glorot_normal", "glorot_uniform", "he_normal", "he_uniform", "identity", "orthogonal", "variance_scaling"]

results_dataframe = MLP_Assembly(optimizer, loss_function, X_train, y_train, X_test, y_test, X_val, y_val, epoch, Nodes, activation, counter, num_lab, b, layer_number)
results_dataframe.to_csv("test_results/{}/{}.csv".format(file_loc, start))
