from pyDeepInsight import ImageTransformer, LogScaler
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D 
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import BatchNormalization
import numpy as np
import pandas as pd
import sklearn
import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import requests
import zipfile
import csv
import scanpy as sc
import argparse
import anndata

#Load data
parser = argparse.ArgumentParser(description='Select dataset')
parser.add_argument('path', type = int)
args = parser.parse_args()
if args.path == 1:
  labels =pd.read_csv("labels_1.csv", names = ["X"])
  data = sc.read("results_1.h5ad")
if args.path == 2:
  labels =pd.read_csv("labels_2.csv", names = ["X"])
  data = sc.read("results_2.h5ad")
num_lab = len(labels)
counter = 0

#perform pca on data
print("The original shape of the data is {}".format(data.shape))
sc.tl.pca(data, svd_solver='arpack')

#split data
X_train, X_test, y_train, y_test = train_test_split(data.X, labels, test_size=0.33, random_state=42)

#scale data
ln = LogScaler()
X_train_norm = ln.fit_transform(X_train)
X_test_norm = ln.transform(X_test)

#split data using pca
#thoughts, tsne versus pca for image extraction
it = ImageTransformer(feature_extractor='pca', 
                      pixels=50, random_state=1701, 
                      n_jobs=-1)
fig = plt.figure(figsize=(5, 5))
_ = it.fit(X_train_norm, plot=True)

fig.savefig('CNN_graphs/test_graph')
