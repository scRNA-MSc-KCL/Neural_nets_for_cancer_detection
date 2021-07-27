"""interdatset----------"""

#step 1; set it up so it works with dataset 2
#step2; set it up so it works with any dataset and including functions. 
import numpy as np
import pandas as pd
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
import csv
import scanpy as sc


#Create output file
#f = open("myfile.txt", "w")
#f.write("Interdataset testing!")
#f.close()

#f = open("myfile.txt", "r")
#print(f.read())
counter = 0

#Unzip files
with zipfile.ZipFile("Dataset1_interdataset.zip", 'r') as zip_ref:
    zip_ref.extractall()

labels =pd.read_csv("Labels.csv")
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
labels = pd.Series(labels)
num_lab = len(pd.unique(labels))
labels = labels.to_numpy()


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

#make labels for neural network catagorical
y_train = to_categorical(y_train, num_lab)
y_test = to_categorical(y_test, num_lab)


#Neural network testing function
def MLP_Assembly(optimizer, loss_function, X_train, y_train, X_test, y_test, epoch, Nodes, activation, counter):
  optimizer_list = []
  loss_function_list = []
  epoch_list = []
  node_1_length_list = []
  activation_layer_1_list = []
  node_2_length_list = []
  activation_layer_2_list = []
  percentage_misclassified = []
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
                optimizer_list.append(o)
                loss_function_list.append(l)
                activation_layer_1_list.append(a1)
                activation_layer_2_list.append(a2)
                epoch_list.append(e)
                node_1_length_list.append(n1)
                node_2_length_list.append(n2)
                net.add(Dense(n2, activation = a2))
                net.add(Dense(5, activation='softmax'))
                net.compile(loss=l, optimizer=o)
                history = net.fit(X_train, y_train,
                            validation_data=(X_test, y_test),
                            epochs=e,batch_size=25)
                outputs = net.predict(X_test)
                labels_predicted= np.argmax(outputs, axis=1)
                y_test_decoded = np.argmax(y_test, axis=1)  # maybe change so you're not doing every time
                misclassified =  (np.sum(labels_predicted != y_test_decoded)/(len(y_test_decoded)))*100
                percentage_misclassified.append(misclassified)
                print("model number", counter)
  print(percentage_misclassified)
  df = pd.DataFrame(list(zip(optimizer_list, loss_function_list, epoch_list, node_1_length_list, activation_layer_1_list, node_2_length_list, activation_layer_2_list, percentage_misclassified)),
                        columns =['optimizer', 'loss_function', "epochs", "node1_length", "activation_layer1", "node2_length" , "activation_layer2", "perceptage_misclassified"])
  return df

#define variables
Nodes = np.arange(10, 2010, 500)
activation = ["tanh", "relu", "sigmoid", "softplus", "softsign", "selu", "elu"]
#activation = ["tanh"]
optimizer = ["SGD", "RMSprop", "Adam", "Adadelta", "Adagrad", "Adamax", "Nadam", "Ftrl"]
#optimizer = ["Nadam"]
epoch = [3]
#epoch = [5]

#loss_function = ["categorical_crossentropy", "poisson","kl_divergence"]
loss_function = ["categorical_crossentropy"]
#consider using custom learning rate
#may or may not get used. See impact on above results
regularizer = ["l1", "l2", "l1_l2"]
kernal_init = ["random_normal", "random_uniform", "truncated_normal", "zeros", "ones", "glorot_normal", "glorot_uniform", "he_normal", "he_uniform", "identity", "orthogonal", "variance_scaling"]

results_dataframe = MLP_Assembly(optimizer, loss_function, X_train, y_train, X_test, y_test, epoch, Nodes, activation, counter)
results_dataframe.to_csv("MLP_Optimization_results_DS1.csv")
