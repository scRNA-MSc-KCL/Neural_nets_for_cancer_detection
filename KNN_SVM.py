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

labels =pd.read_csv("labels_3.csv", names = ["X"])
data = sc.read("results_3.h5ad")
neigh = NearestNeighbors(n_neighbors=2000)
neigh.fit(data)
NearestNeighbors(n_neighbors=1)
print(neigh.kneighbors([[1., 1., 1.]]))
(array([[0.5]]), array([[2]]))#
print(neigh.kneighbors([[1., 1., 1.]]))
(array([[0.5]]), array([[2]]))
