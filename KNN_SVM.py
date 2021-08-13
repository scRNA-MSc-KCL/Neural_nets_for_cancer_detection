import pandas as pd
import scanpy as sc
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import time
import argparse
import anndata
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier

start = time.time()
labels =pd.read_csv("labels_1.csv", names = ["x"])
data = sc.read("results_1.h5ad")
X_train, X_test, y_train, y_test = train_test_split(data.X, labels, test_size=0.1)
neigh = KNeighborsClassifier(n_neighbors=200)
neigh.fit(X_train, y_train)
y_train = y_train.reset_index()
mode_list = []
for i in X_test:
  neighbours = neigh.kneighbors([i], return_distance = False)
  SVM_data = X_train[neighbours]
  indices = neighbours.tolist()
  indices = indices[0]
  SVM_labels = y_train['x'][indices]
  mode_list.append(SVM_labels.value_counts().idxmax())
  SVM_labels = SVM_labels.to_list()

knn_accuracy = 0
knnsvm_accuracy = 0
for i in range(len(y_train)):
  if y_train['x'][i] == mode_list[i]:
    knn_accuracy += 1
  if y_train['x'][i] == SVM_labels[i]:
    knnsvm_accuracy += 1
knn_accuracy = (knn_accuracy/len(y_train))*100
    
end = time.time()
print("The time taken to complete this program was {}".format(end - start))
