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

labels =pd.read_csv("labels_1.csv", names = ["X"])
data = sc.read("results_1.h5ad")

X_train, X_test, y_train, y_test = train_test_split(data.X, labels, test_size=0.2, random_state=42)
neigh = KNeighborsClassifier(n_neighbors=1000)
neigh.fit(X_train, y_train)
y_train = y_train.reset_index()
knn_list = []
SVM_list = []

for i in X_test:
  neighbours = neigh.kneighbors([i], return_distance = False)
  SVM_data = X_train[neighbours]
  indices = neighbours.tolist()
  indices = indices[0]
  SVM_labels = y_train['x'][indices]
  knn_list.append(SVM_labels.value_counts().idxmax())
  SVM_labels = SVM_labels.to_list()
  Classifier = sklearn.svm.SVC(kernel = "linear")
  #print(SVM_data[0])
  Classifier.fit(SVM_data[0], SVM_labels)
  y_pred = Classifier.predict([i])
  SVM_list.append(y_pred)


print("The accuracy using knn is {}".format(mode_list))
print("The accuract using SVM is {}".format(y_pred))
