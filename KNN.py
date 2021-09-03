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
  data = sc.read("results_2.h5ad
#Dataset 3
if args.path == 3:
  labels =pd.read_csv("labels_3.csv", names = ["X"])
  data = sc.read("results_3.h5ad")
if args.path == 4:
  labels =pd.read_csv("labels_4.csv", names = ["X"])
  data = sc.read("results_4.h5ad")

neighbours = [1,3,5,7,9]

#Create model for each number of neighbours
for n in neighbours:
  X_train, X_test, y_train, y_test = train_test_split(data.X, labels, test_size=0.2, random_state = 42)
  neigh = KNeighborsClassifier(n_neighbors=n)
  neigh.fit(X_train, y_train)
  print(neigh.score(X_test, y_test))
    
end = time.time()
print("The time taken to complete this program was {}".format(end - start))
