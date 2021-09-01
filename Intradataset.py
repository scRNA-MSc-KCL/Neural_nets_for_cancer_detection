import pandas as pd
import scanpy as sc
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time
import argparse
import anndata
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical

labels_10x =pd.read_csv("labels_5.csv", names = ["X"])
labels_10x = labels_10x['X']
data_10x = sc.read("results_5.h5ad")
labels_celseq =pd.read_csv("labels_6.csv", names = ["X"])
labels_celseq = labels_celseq['X']
print(len(labels_celseq))
data_celseq = sc.read("results_6.h5ad")

#K Nearest Neighbours

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(data_10x.X, labels_10x)
print("The intradataset nearest neighbour score;", neigh.score(data_celseq.X, labels_celseq))

#SVM

Classifier = sklearn.svm.SVC(kernel = 'rbf')
Classifier.fit(data_10x.X, labels_10x)
print("The intradataset SVM score {}".format(Classifier.score(data_celseq.X, labels_celseq)))
                                                              
#Multilayer Perceptron
X_train, X_val, y_train, y_val = train_test_split(data_10x.X, labels_10x, test_size=0.2, random_state=42)
#make labels for neural network catagorical
y_train = to_categorical(y_train, 5)
y_test = to_categorical(labels_celseq, 5)
y_val = to_categorical(y_val, 5)

#Build Network
net = Sequential()
net.add(Dense(500, activation = 'relu', input_shape = (data_10x.n_vars,)))
net.add(Dense(5, activation='softmax'))
net.compile(loss="categorical_crossentropy", optimizer='Adam')
history = net.fit(X_train, y_train,validation_data=(X_val, y_val),epochs=7,batch_size=50)
outputs = net.predict(data_celseq.X)
labels_predicted= np.argmax(outputs, axis=1)
y_test_decoded = np.argmax(y_test, axis=1)  # maybe change so you're not doing every time
correctly_classified =  (np.sum(labels_predicted == y_test_decoded)/(len(y_test_decoded)))*100
print("The intradataset MLP score {}".format(correctly_classified))
                                                              
