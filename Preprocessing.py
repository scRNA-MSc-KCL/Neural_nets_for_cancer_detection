import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import requests
import zipfile
import csv
import scanpy as sc
import argparse
import anndata
import os

#3 is UMI raw
#4 is normalized
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
  #filename = 'GSE131907_Lung_Cancer_raw_UMI_matrix.txt.gz'
  #os.system('gunzip ' + filename)
  #Unzip files
  #txt_file = "GSE131907_Lung_Cancer_raw_UMI_matrix.txt"
  #csv_file = "GSE131907_Lung_Cancer_raw_UMI_matrix.csv"
  #in_txt = csv.reader(open(txt_file, "r"), delimiter = '\t')
  #out_csv = csv.writer(open(csv_file, 'w'))
  #out_csv.writerows(in_txt)
  labels =pd.read_csv("GSE131907_Lung_Cancer_cell_annotation.txt", sep = "\t")
  data = sc.read_csv("GSE131907_Lung_Cancer_normalized_log2TPM_matrix.csv") #29634 x 208506
  labels = labels["Cell_type"]
if args.path == 4:
  labels =pd.read_csv("GSE131907_Lung_Cancer_cell_annotation.txt", sep = "\t")
  data = sc.read_csv("GSE131907_Lung_Cancer_raw_UMI_matrix.csv")
  labels = labels["Cell_type"]

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
print("The original shape of the data1 is {}".format(data))
#normalize data
sc.pp.normalize_total(data, target_sum=10000)
print("A")
#logarithmize data
sc.pp.log1p(data)
print("b")

#select highly variable genes
sc.pp.highly_variable_genes(data, n_top_genes=1000)
print("c")
data = data[:, data.var.highly_variable]

print("The final shape of the data1 is {}".format(data.shape))
print("d")
#sc.tl.pca(data, svd_solver='arpack')
#print("The shape after performing pca is {}".format(data.shape))

#sc.pp.neighbors(data, n_neighbors=10, n_pcs=40)
#print("The shape after doing neightbours things is {}".format(data.shape))
#sc.tl.leiden(data)
#print("The shape after doing leiden thing is {}".format(data.shape))


data.write("adata_obj_{}.anndata".format(args.path))
print("E")
np.savetxt("labels_{}.csv".format(args.path), labels, delimiter=",")

