import pandas as pd
import scanpy as sc
import numpy as np
import sklearn
import tensorflow
import keras
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import anndata

data = sc.read_csv("human_cell_atlas/krasnow_hlca_facs_counts.csv")  #58683 x 9409
#data_10x = sc.read_csv("human_cell_atlas/krasnow_hlca_10x_UMIs.csv") #26485 x 65662
labels = pd.read_csv("human_cell_atlas/krasnow_hlca_facs_metadata.csv") #9409 x 141
#labels_10x = pd.read_csv("human_cell_atlas/krasnow_hlca_10x_metadata.csv") #65662 x 21


#print("facs counts", data)
#print("UMI counts", data_10x)
#print("facs labels", labels)
#print("10x labels", labels_10x)

#labels
labels = labels["free_annotation"]
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
labels = pd.Series(labels)
num_lab = len(pd.unique(labels))
print("number of labels", num_lab)
labels = labels.to_numpy()

#read data
#data = anndata.AnnData.transpose(data)
print("The original shape of the data is {}".format(data))

#normalize data
sc.pp.normalize_total(data, target_sum=10000)
#logarithmize data
sc.pp.log1p(data)

#select highly variable genes
sc.pp.highly_variable_genes(data, n_top_genes=1000)
data = data[:, data.var.highly_variable]

print("The original shape of the data is {}".format(data.shape))
sc.tl.pca(data, svd_solver='arpack')
print("The shape after regressing out the data is {}".format(data.shape))

#create training and test sets
X_train, X_test, y_train, y_test = train_test_split(data.X, labels, test_size=0.2, random_state=42)

#make labels for neural network catagorical
y_train = to_categorical(y_train, num_lab)
y_test = to_categorical(y_test, num_lab)


counter = 0

#Neural network testing function
def MLP_Assembly(optimizer, loss_function, X_train, y_train, X_test, y_test, epoch, Nodes, activation, counter, num_lab):
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
                net.add(Dense(num_lab, activation='softmax'))
                net.compile(loss=l, optimizer=o)
                history = net.fit(X_train, y_train,
                            validation_data=(X_test, y_test),
                            epochs=e,batch_size=25)
                outputs = net.predict(X_test)
                labels_predicted= np.argmax(outputs, axis=1)
                y_test_decoded= np.argmax(y_test, axis=1)
                misclassified =  (np.sum(labels_predicted != y_test_decoded)/(len(y_test_decoded)))*100
                percentage_misclassified.append(misclassified)
                print("model number", counter)
  print(percentage_misclassified)
  df = pd.DataFrame(list(zip(optimizer_list, loss_function_list, epoch_list, node_1_length_list, activation_layer_1_list, node_2_length_list, activation_layer_2_list, percentage_misclassified)),
                        columns =['optimizer', 'loss_function', "epochs", "node1_length", "activation_layer1", "node2_length" , "activation_layer2", "perceptage_misclassified"])
  return df
      


#define variables
Nodes = np.arange(10, 2010, 1000)
#activation = ["tanh", "relu", "sigmoid", "softplus", "softsign", "selu", "elu"]
activation = ["tanh"]
#optimizer = ["SGD", "RMSprop", "Adam", "Adadelta", "Adagrad", "Adamax", "Nadam", "Ftrl"]
optimizer = ["Nadam"]
epoch = [3]
#epoch = [5]

#loss_function = ["categorical_crossentropy", "poisson","kl_divergence"]
loss_function = ["categorical_crossentropy"]
#consider using custom learning rate
#may or may not get used. See impact on above results
regularizer = ["l1", "l2", "l1_l2"]
kernal_init = ["random_normal", "random_uniform", "truncated_normal", "zeros", "ones", "glorot_normal", "glorot_uniform", "he_normal", "he_uniform", "identity", "orthogonal", "variance_scaling"]

results_dataframe = MLP_Assembly(optimizer, loss_function, X_train, y_train, X_test, y_test, epoch, Nodes, activation, counter, num_lab)
results_dataframe.to_csv("MLP_Optimization_results_DS2.csv")





