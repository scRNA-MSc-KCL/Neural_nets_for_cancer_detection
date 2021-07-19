"""interdatset----------"""

#step 1; set it up so it works with dataset 2
#step2; set it up so it works with any dataset and including functions. 

import numpy as np
import pandas as pd
import scanpy as sc
import sklearn
import tensorflow
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


#Create output file
#f = open("myfile.txt", "w")
#f.write("Interdataset testing!")
#f.close()

#f = open("myfile.txt", "r")
#print(f.read())

#Unzip files
with zipfile.ZipFile("Dataset1_interdataset.zip", 'r') as zip_ref:
    zip_ref.extractall()

#read lables
labels =pd.read_csv("Labels.csv")
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
labels = pd.Series(labels)
num_lab = len(pd.unique(labels))
print("number of labels", num_lab)
labels = labels.to_numpy()
labels = to_categorical(labels, num_lab)

#read data
data = sc.read_csv("Combined_10x_CelSeq2_5cl_data.csv")

#normalize data
sc.pp.normalize_total(data, target_sum=10000)
#logarithmize data
sc.pp.log1p(data)

#select highly variable genes
sc.pp.highly_variable_genes(data, n_top_genes=1000)
data = data[:, data.var.highly_variable]

#create training and test sets
X_train, X_test, y_train, y_test = train_test_split(data.X, labels, test_size=0.2, random_state=42)

#Neural network testing function
def MLP_Assembly(optimizer, loss_function, X_train, y_train, X_test, y_test, epoch, Nodes, activation):
  counter = 0
  MLP_results = open('MLP results.txt', 'w')
  for o in optimizer:
    for l in loss_function:
      for e in epoch:
        net = Sequential()
        for n1 in Nodes:
          for a1 in activation:
            net.add(Dense(n1, activation = a1, input_shape = (data.n_vars,)))
            for n2 in Nodes:
              for a2 in activation:
                counter += 1
                net.add(Dense(n2, activation = a2))
                net.add(Dense(7, activation='softmax'))
                net.compile(loss=l, optimizer=o)
                history = net.fit(X_train, y_train,
                            validation_data=(X_test, y_test),
                            epochs=e,batch_size=25)
                outputs = net.predict(X_test)
                labels_predicted= np.argmax(outputs, axis=1)
                misclassified =  np.sum(labels_predicted != y_test)
                MLP_results.write("counter {}".format(counter))
                MLP_results.write("\n")
                MLP_results.write("activation input later = {}, nodes in input layer = {}, nodes in output layer = {}, activation output later = {}".format(a1, n1, n2, a2))
                MLP_results.write("\n")
                MLP_results.write('Percentage misclassified = {}'.format(100*misclassified/y_test.size))
                MLP_results.write("\n")
                MLP_results.write("optimizer = {}, loss function = {}, epoch = {}".format(o, l, e))
                MLP_results.write("\n")
                MLP_results.write("\n")
    MLP_results.close()

#define variables
Nodes = np.arange(500, 1000, 500)
#activation = ["tanh", "relu", "sigmoid", "softplus", "softsign", "selu", "elu"]
activation = ["tanh"]
#optimizer = ["SGD", "RMSprop", "Adam", "Adadelta", "Adagrad", "Adamax", "Nadam", "Ftrl"]
optimizer = ["Nadam"]
epoch = [3]

#loss_function = ["categorical_crossentropy","sparse_categorical_crossentropy", "poisson","kl_divergence"]
loss_function = ["categorical_crossentropy","sparse_categorical_crossentropy"]
#consider using custom learning rate
#may or may not get used. See impact on above results
regularizer = ["l1", "l2", "l1_l2"]
kernal_init = ["random_normal", "random_uniform", "truncated_normal", "zeros", "ones", "glorot_normal", "glorot_uniform", "he_normal", "he_uniform", "identity", "orthogonal", "variance_scaling"]

MLP_Assembly(optimizer, loss_function, X_train, y_train, X_test, y_test, epoch, Nodes, activation)
