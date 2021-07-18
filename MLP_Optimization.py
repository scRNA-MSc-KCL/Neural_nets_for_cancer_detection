"""interdatset----------"""

#step 1; set it up so it works with dataset 2
#step2; set it up so it works with any dataset and including functions. 

import numpy as np
import pandas as pd
import scanpy as sc
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
import requests
import zipfile

#Unzip files
with zipfile.ZipFile("Dataset1_interdataset.zip", 'r') as zip_ref:
    zip_ref.extractall()

#read lables
labels =pd.read_csv("Labels.csv")
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
labels = pd.Series(labels)
num_labs = len(pd.unique(labels))
print("number of labels", num_lab)
labels = labels.to_numpy()
labels = to_categorical(labels, num_lab)

#read data
data = sc.read_csv("Combined_10x_CelSeq2_5cl_data.csv")

#normalize data
sc.pp.normalize_total(data, target_sum=10000)
#logarithmize data
sc.pp.log1p(data)

#select highly variable genes
sc.pp.highly_variable_genes(data, n_top_genes=1000)
data = data[:, data.var.highly_variable]

#create training and test sets
X_train, X_test, y_train, y_test = train_test_split(data.X, labels, test_size=0.2, random_state=42)

