import pandas as pd
import scanpy as sc
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import anndata
import matplotlib.pyplot as plt
import requests
import zipfile
import csv
import scanpy as sc
import argparse
import time
import os
from sklearn.model_selection import KFold
from pycm import *
import seaborn
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from numpy import savetxt
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier

accuracy_list = []
counter = 0

parser = argparse.ArgumentParser(description='Select dataset')
parser.add_argument('path', type = int)
start = time.time()
args = parser.parse_args()
#Dataset 1
if args.path == 1:
  labels =pd.read_csv("labels_1.csv", names = ["X"])
  data = sc.read("results_1.h5ad")
  n = 3
  file_loc = "DS1/KNN"
#Dataset 2
if args.path == 2:
  labels =pd.read_csv("labels_2.csv", names = ["X"])
  data = sc.read("results_2.h5ad")
  file_loc = "DS2/KNN"
  n = 9
#Dataset 3
if args.path == 4:
  labels =pd.read_csv("labels_4.csv", names = ["X"])
  data = sc.read("results_4.h5ad")
  file_loc = "DS4/KNN"
  n = 5
  
#Create output directory 
path = os.getcwd()
path = os.path.join(path, "test_results/{}/{}".format(file_loc,start))
try:
  os.makedirs(path)
except OSError:
  print("Creation of the directory %s failed" % path)
else:
  print("Successfully created the directory %s" % path)

num_lab = len(labels["X"].unique())
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

#Split training data
kf = KFold(n_splits=5, shuffle = True)
for train_index, test_index in kf.split(data.X):
  X_train, X_test = data.X[train_index], data.X[test_index]
  y_train, y_test = labels[train_index], labels[test_index]
  
  #Create model
  neigh = KNeighborsClassifier(n_neighbors=n)
  neigh.fit(X_train, y_train)
  
  #Test model
  print(neigh.score(X_test, y_test))
  print("the classification result with the current settings  is {}".format(neigh.score(X_test, y_test)))
  y_pred = neigh.predict(X_test)
  with open('test_results/{}/{}/summary{}.txt'.format(file_loc, start, counter), 'w') as fr:
    fr.write("precision score: {}".format(precision_score(y_test, y_pred, average=None)))
    fr.write("recall score: {} ".format(recall_score(y_test, y_pred, average=None)))
    fr.write("accuracy: {}".format(neigh.score(X_test, y_test)))
  print(precision_score(y_test, y_pred, average=None))
  print(recall_score(y_test, y_pred, average=None))
  savetxt("test_results/{}/{}/{}_ypred.csv".format(file_loc, start, counter), y_pred, delimiter=',')
  savetxt("test_results/{}/{}/{}_ytrue.csv".format(file_loc, start, counter), y_test, delimiter=',')
  counter +=1
  
  
end = time.time()
print("The time taken to complete this program was {}".format(end - start))
