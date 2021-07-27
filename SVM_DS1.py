import pandas as pd
import scanpy as sc
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import anndata


#Unzip files
#with zipfile.ZipFile("Dataset1_interdataset.zip", 'r') as zip_ref:
#    zip_ref.extractall()

labels =pd.read_csv("Labels.csv")
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
labels = pd.Series(labels)
num_lab = len(pd.unique(labels))
labels = labels.to_numpy()


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

#make labels for neural network catagorical
y_train = to_categorical(y_train, num_lab)
y_test = to_categorical(y_test, num_lab)
                                                    
#create training and test sets
X_train, X_test, y_train, y_test = train_test_split(data.X, labels, test_size=0.2, random_state=42)

SVM_list = ["linear", "poly", "rbf", "sigmoid"]

# build classifier
                                                    
for i in SVM_list:
  Classifier = sklearn.svm.SVC(kernel = i)
  Classifier.fit(X_train, y_train)
  print("the classification result with the current settings and a {} kernal is {}".format(i, Classifier.score(X_test, y_test)))
