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
import matplotlib.pyplot as plt
import requests
import zipfile
import csv
import scanpy as sc
import argparse
import anndata
import time
import os
from sklearn.model_selection import KFold
import pickle

accuracy_list = []
counter = 0

parser = argparse.ArgumentParser(description='Select dataset')
parser.add_argument('path', type = int)
start = time.time()

args = parser.parse_args()
if args.path == 1:
  labels =pd.read_csv("labels_1.csv", names = ["X"])
  data = sc.read("results_1.h5ad")
  file_loc = "DS1/SVM"
if args.path == 2:
  labels =pd.read_csv("labels_2.csv", names = ["X"])
  data = sc.read("results_2.h5ad")
  file_loc = "DS2/SVM"
if args.path == 4:
  labels =pd.read_csv("labels_4.csv", names = ["X"])
  data = sc.read("results_4.h5ad")
  file_loc = "DS4/SVM"
  
path = os.getcwd()
path = os.path.join(path, "test_results/{}/{}".format(file_loc,start))
try:
  os.makedirs(path)
except OSError:
  print("Creation of the directory %s failed" % path)
else:
  print("Successfully created the directory %s" % path)

num_lab = len(labels["X"].unique())

#Split training data
kf = KFold(n_splits=5)
for train_index, test_index in kf.split(data.X):
  X_train, X_test = X_train[train_index], X_train[test_index]
  y_train, y_test = y_train['X'][train_index], y_train['X'][test_index]
  Classifier = sklearn.svm.SVC(kernel = "rbf")
  Classifier.fit(X_train, y_train)
  print("classifier {} trained".format(i))
  print("the classification result with the current settings and a {} kernal is {}".format(i, Classifier.score(X_test, y_test)))
  
  
end = time.time()
print("The time taken to complete this program was {}".format(end - start))
