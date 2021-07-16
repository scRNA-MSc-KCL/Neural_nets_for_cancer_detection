"""interdatset----------"""

#step 1; set it up so it works with dataset 2
#step2; set it up so it works with any dataset and including functions. 

import numpy as np
import pandas as pd
"""import scanpy as sc
import sklearn
import tensorflow
import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import requests"""
import zipfile

#Unzip files
with zipfile.ZipFile("Dataset1_interdataset.zip", 'r') as zip_ref:
    zip_ref.extractall()
    
"""labels =pd.read_csv("Labels.csv")
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
data = sc.read_csv("Combined_10x_CelSeq2_5cl_data.csv")
length_dset = len(data)
data.obs["annot_index"] = range(length_dset)
#add a sequential index that will be used to cut labels at a later date
sc.pp.normalize_total(data, target_sum=10000)
sc.pp.log1p(data)
labels = pd.Series(labels)
labels = labels.loc[data.obs["annot_index"]]
num_labs = len(pd.unique(labels))
print("number of labels", num_labs)

sc.pp.highly_variable_genes(data, n_top_genes=1000)
data = data[:, data.var.highly_variable]
print("The shape after regressing out the data is {}".format(data.shape))
sc.tl.pca(data, svd_solver='arpack')
print("The shape after regressing out the data is {}".format(data.shape))
labels = labels.to_numpy()
labels = to_categorical(labels, 5)
data = data.X

X_train, X_test, y_train, y_test = train_test_split(data.X, labels, test_size=0.2, random_state=42)
y_test = y_test.to_numpy()
y_train = y_train.to_numpy()
y_train = to_categorical(y_train, 5)
y_test = to_categorical(y_test, 5)"""
