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
  SVM_list.append(y_pred[0])
  

print(knn_list)
print(SVM_list)
print(y_train)
print(len(knn_list))
print(len(SVM_list))
print(len(y_train))

#knn_accuracy = 0
#knnsvm_accuracy = 0
#for i in range(len(SVM_labels)):
#  if SVM_labels[i] == knn_list[i]:
#    knn_accuracy += 1
#  if SVM_labels[i] == SVM_labels[i]:
#    knnsvm_accuracy += 1
#knn_accuracy = (knn_accuracy/len(y_train))*100
    
end = time.time()
print("The time taken to complete this program was {}".format(end - start))
