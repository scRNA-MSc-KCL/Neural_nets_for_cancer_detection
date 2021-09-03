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

start = time.time()

#Load data
parser = argparse.ArgumentParser(description='Select dataset')
parser.add_argument('path', type = int)
args = parser.parse_args()
#Dataset 1
if args.path == 1:
  labels =pd.read_csv("labels_1.csv", names = ["X"])
  data = sc.read("results_1.h5ad")
#Dataset 2
if args.path == 2:
  labels =pd.read_csv("labels_2.csv", names = ["X"])
  data = sc.read("results_2.h5ad")
#Dataset 3
if args.path == 4:
  labels =pd.read_csv("labels_4.csv", names = ["X"])
  data = sc.read("results_4.h5ad")
  
#encode labels
num_lab = len(labels["X"].unique())
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)  

#create training and test sets
X_train, X_test, y_train, y_test = train_test_split(data.X, labels, test_size=0.2)


SVM_list = ["linear", "poly", "rbf", "sigmoid"]

# build classifier                                                  
for i in SVM_list:
  Classifier = sklearn.svm.SVC(kernel = i)
  Classifier.fit(X_train, y_train)
  print("classifier {} trained".format(i))
  print("the classification result with the current settings and a {} kernal is {}".format(i, Classifier.score(X_test, y_test)))
  
#print results  
end = time.time()
print("The time taken to complete this program was {}".format(end - start))
