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

def unzip_file(filename):
  os.system('gunzip ' + filename)
  
def unzip_gz_file(txt_file_name, csv_file_name):
  in_txt = csv.reader(open(txt_file_name, "r"), delimiter = '\t')
  out_csv = csv.writer(open(csv_file_name, 'w'))
  out_csv.writerows(in_txt)

def label_adaption(labels):
  label_encoder = LabelEncoder()
  labels = label_encoder.fit_transform(labels)
  labels = pd.Series(labels)
  num_lab = len(pd.unique(labels))
  print("number of labels", num_lab)
  labels = labels.to_numpy()
  return labels  
  

#3 is UMI raw
#4 is normalized

parser = argparse.ArgumentParser(description='Select dataset')
parser.add_argument('path', type = int)

args = parser.parse_args()
if args.path == 1:
  #Unzip files - for dataset 1
#with zipfile.ZipFile("Dataset1_interdataset.zip", 'r') as zip_ref:
#    zip_ref.extractall()
  labels =pd.read_csv("Labels.csv")
  data = sc.read_csv("Combined_10x_CelSeq2_5cl_data.csv")
  results = 'results_1.h5ad'
if args.path == 2:
  data = sc.read_csv("human_cell_atlas/krasnow_hlca_10x_UMIs.csv") #26485 x 65662
  data = anndata.AnnData.transpose(data)
  #labels = pd.read_csv("human_cell_atlas/krasnow_hlca_facs_metadata.csv") #9409 x 141
  ##data = sc.read_csv("human_cell_atlas/krasnow_hlca_facs_counts.csv")  #58683 x 9409
  labels = pd.read_csv("human_cell_atlas/krasnow_hlca_10x_metadata.csv") #65662 x 21
  labels = labels["free_annotation"]
  results = 'results_2.h5ad'
if args.path == 3:
  #Unzip files - for dataset 3
  unzip_file('GSE131907_Lung_Cancer_raw_UMI_matrix.txt.gz')
  #Unzip files
  unzip_gz_file("GSE131907_Lung_Cancer_raw_UMI_matrix.txt", "GSE131907_Lung_Cancer_raw_UMI_matrix.csv")
  labels =pd.read_csv("GSE131907_Lung_Cancer_cell_annotation.txt", sep = "\t")
  data = sc.read_csv("GSE131907_Lung_Cancer_raw_UMI_matrix.csv") #29634 x 208506
  data = anndata.AnnData.transpose(data)
  labels = labels["Cell_type"]
  results = 'results_3.h5ad'
if args.path == 4:
  #unzip gz files
  #import tarfile
  #tf = tarfile.open("GSE131508_RAW.tar")
  #tf.extractall()
  #Unzip files
  #cherry positive
  #unzip_file('GSM3783354_4T1_CherryPositive_RawCounts.txt.gz')
  #unzip_file('GSM3783356_4T1_CherryNegative_RawCounts.txt.gz')
  #unzip_gz_file("GSM3783354_4T1_CherryPositive_RawCounts.txt", "GSM3783354_4T1_CherryPositive_RawCounts.csv")
  #cherry negative
  #unzip_file('GSM3783356_4T1_CherryNegative_RawCounts.txt.gz')
  #unzip_gz_file("GSM3783356_4T1_CherryNegative_RawCounts.txt", "GSM3783356_4T1_CherryNegative_RawCounts.csv")
  data_pos = pd.read_csv("GSM3783354_4T1_CherryPositive_RawCounts.csv")
  data_neg = pd.read_csv("GSM3783356_4T1_CherryNegative_RawCounts.csv")
  data_pos = data_pos.drop(["Unnamed: 0"], axis = 1)
  data_neg = data_neg.drop(["Unnamed: 0"], axis = 1)
  data_pos = data_pos.set_index("Gene_Symbol")
  data_neg = data_neg.set_index("Gene_Symbol")
  data = pd.concat([data_pos, data_neg], axis = 1)
  data = data.fillna(0)
  data = sc.AnnData(data)
  data.var_names_make_unique() 
  data.obs_names_make_unique()
  data = anndata.AnnData.transpose(data)
  results = 'results_4.h5ad'
  l_pos = len(data_pos.columns)
  l_neg = len(data_neg.columns)
  label_pos = ["Cherry Positive"]*l_pos
  label_neg = ["Cherry Negative"]*l_neg
  labels = label_pos + label_neg

labels = label_adaption(labels)

#read data
print("The original shape of the data1 is {}".format(data))
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

data.write(results)
print("data shape", data)
np.savetxt("labels_{}.csv".format(args.path), labels, delimiter=",")
