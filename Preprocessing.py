import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import requests
import zipfile
import csv
import scanpy as sc
import argparse
import anndata
import os


parser = argparse.ArgumentParser(description='Select dataset')
parser.add_argument('path', type = int)

args = parser.parse_args()
if args.path == 1:
  labels =pd.read_csv("Labels.csv")
  data = sc.read_csv("Combined_10x_CelSeq2_5cl_data.csv")
if args.path == 2:
  data = sc.read_csv("human_cell_atlas/krasnow_hlca_10x_UMIs.csv") #26485 x 65662
  data = anndata.AnnData.transpose(data)
  #labels = pd.read_csv("human_cell_atlas/krasnow_hlca_facs_metadata.csv") #9409 x 141
  ##data = sc.read_csv("human_cell_atlas/krasnow_hlca_facs_counts.csv")  #58683 x 9409
  labels = pd.read_csv("human_cell_atlas/krasnow_hlca_10x_metadata.csv") #65662 x 21
  labels = labels["free_annotation"]
if args.path == 3:
  #Unzip files - for dataset 3
  labels =pd.read_csv("GSE131907_Lung_Cancer_cell_annotation.txt", sep = "\t")
  data = sc.read_csv("GSE131907_Lung_Cancer_raw_UMI_matrix.csv")

#Unzip files - for dataset 1
#with zipfile.ZipFile("Dataset1_interdataset.zip", 'r') as zip_ref:
#    zip_ref.extractall()

def label_adaption(labels):
  label_encoder = LabelEncoder()
  labels = label_encoder.fit_transform(labels)
  labels = pd.Series(labels)
  num_lab = len(pd.unique(labels))
  print("number of labels", num_lab)
  labels = labels.to_numpy()
  return labels

labels = label_adaption(labels)

#read data
print("The original shape of the data is {}".format(data))

#normalize data
sc.pp.normalize_total(data, target_sum=10000)
#logarithmize data
sc.pp.log1p(data)

#select highly variable genes
sc.pp.highly_variable_genes(data, n_top_genes=1000)
data = data[:, data.var.highly_variable]

print("The final shape of the data is {}".format(data.shape))
#sc.tl.pca(data, svd_solver='arpack')
#print("The shape after performing pca is {}".format(data.shape))

#sc.pp.neighbors(data, n_neighbors=10, n_pcs=40)
#print("The shape after doing neightbours things is {}".format(data.shape))
#sc.tl.leiden(data)
#print("The shape after doing leiden thing is {}".format(data.shape))


data.write("adata_obj_{}".format(args.path))
np.savetxt("labels_{}.csv".format(args.path), labels, delimiter=",")
labels.to_csv("labels_{}.csv".format(args.path))

