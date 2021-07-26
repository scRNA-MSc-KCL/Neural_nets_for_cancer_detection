import pandas as pd
import scanpy as sc
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import anndata

#read data
data = sc.read_csv("human_cell_atlas/krasnow_hlca_10x_UMIs.csv") #26485 x 65662
data = anndata.AnnData.transpose(data)
labels = pd.read_csv("human_cell_atlas/krasnow_hlca_10x_metadata.csv") #65662 x 21
print("The original shape of the data is {}".format(data))

#labels
labels = labels["free_annotation"]
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
labels = pd.Series(labels)
num_lab = len(pd.unique(labels))
print("number of labels", num_lab)
labels = labels.to_numpy()


#normalize data
sc.pp.normalize_total(data, target_sum=10000)
#logarithmize data
sc.pp.log1p(data)

#select highly variable genes
sc.pp.highly_variable_genes(data, n_top_genes=1000)
data = data[:, data.var.highly_variable]

print("The original shape of the data is {}".format(data.shape))
sc.tl.pca(data, svd_solver='arpack')
print("The shape after performing pca is {}".format(data.shape))
                                                    
#create training and test sets
X_train, X_test, y_train, y_test = train_test_split(data.X, labels, test_size=0.2, random_state=42)

# build classifier
                                                    
y_test = y_test.to_numpy()
y_train = y_train.to_numpy()
Classifier = sklearn.svm.SVC(kernel = "linear")
Classifier.fit(X_train, y_train)
print("the classification result with the current settings is ",Classifier.score(X_test, y_test))


