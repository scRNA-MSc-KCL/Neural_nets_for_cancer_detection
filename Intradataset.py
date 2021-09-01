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

labels_10x =pd.read_csv("labels_5.csv", names = ["X"])
labels_10x = labels_10x['X']
data_10x = sc.read("results_5.h5ad")
labels_celseq =pd.read_csv("labels_6.csv", names = ["X"])
labels_celseq = labels_celseq['X']
data_celseq = sc.read("results_6.h5ad")

#K Nearest Neighbours

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(data_celseq.X, labels_celseq)
print("The intradataset nearest neighbour score;", neigh.score(data_10x.X, labels_10x))
