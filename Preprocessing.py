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

#Note; this program assumes that files have been unzipped and are stored in a local repository ~/Original_Data/
#Helper functions that can be used to unzip files
def unzip_file(filename):
  os.system('gunzip ' + filename)
  
def unzip_gz_file(txt_file_name, csv_file_name):
  in_txt = csv.reader(open(txt_file_name, "r"), delimiter = '\t')
  out_csv = csv.writer(open(csv_file_name, 'w'))
  out_csv.writerows(in_txt)

#Helper function for parsing label data
def label_adaption(labels):
  label_encoder = LabelEncoder()
  labels = label_encoder.fit_transform(labels)
  labels = pd.Series(labels)
  labels.to_csv("labels_unencoded_2.csv")
  num_lab = len(pd.unique(labels))
  print("number of labels", num_lab)
  labels = labels.to_numpy()
  return labels  
  
#Load data

parser = argparse.ArgumentParser(description='Select dataset')
parser.add_argument('path', type = int)
args = parser.parse_args()

#Dataset 1
if args.path == 1:
  labels =pd.read_csv("Original_data/Labels.csv")
  data = sc.read_csv("Original_data/Combined_10x_CelSeq2_5cl_data.csv")
  results = 'results_1.h5ad'
  
#Dataset 2
if args.path == 2:
  data = sc.read_csv("Original_data/human_cell_atlas/krasnow_hlca_10x_UMIs.csv") 
  data = anndata.AnnData.transpose(data)
  labels = pd.read_csv("Original_data/human_cell_atlas/krasnow_hlca_10x_metadata.csv")
  labels = labels["free_annotation"]
  labels.to_csv("labels_2_unencoded")
  results = 'results_2.h5ad'
  
#Dataset 3
if args.path == 4:
  data_pos = pd.read_csv("Original_data/GSM3783354_4T1_CherryPositive_RawCounts.csv")
  data_neg = pd.read_csv("Original_data/GSM3783356_4T1_CherryNegative_RawCounts.csv")
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
  l_pos = len(data_pos.columns)
  l_neg = len(data_neg.columns)
  label_pos = ["Cherry Positive"]*l_pos
  label_neg = ["Cherry Negative"]*l_neg
  labels = label_pos + label_neg
  pd.Series(labels).to_csv("labels_4_unencoded")
  results = 'results_4.h5ad'
  
#Intradata Splits
#Dataset 1 SmartSeq
if args.path == 5:
  labels =pd.read_csv("Original_data/Labels_10x_5cl.csv")
  data = sc.read_csv("Original_data/10x_5cl_data.csv")
  results = 'results_5.h5ad'
#Dataset 1 CelSeq
if args.path == 6:
  labels =pd.read_csv("Original_data/Labels_CelSeq2_5cl.csv")
  data = sc.read_csv("Original_data/CelSeq2_5cl_data.csv")
  results = 'results_6.h5ad'

#Adjust labels
labels = label_adaption(labels)
print("The original shape of the data1 is {}".format(data))

#Pipeline 1 - filtering genes based on highly variable counts
if args.path == 1 or args.path == 4 or args.path == 5 or args.path == 6: 
  #read data
  sc.pp.filter_genes(data, min_cells=1)
  #normalize data
  sc.pp.normalize_total(data, target_sum=10000)
  #logarithmize data
  sc.pp.log1p(data)
  #select highly variable genes
  sc.pp.highly_variable_genes(data, n_top_genes=2000)
  data = data[:, data.var.highly_variable]
  print("The final shape of the data is {}".format(data.shape))

#Pipeline 2 - filtering genes based on summary statistics
if args.path == 2 or args.path == 3: 
  #read data
  sc.pp.filter_genes(data, min_cells=5)
  #normalize data
  sc.pp.normalize_total(data, target_sum=10000)
  #logarithmize data
  sc.pp.log1p(data)
  #select highly variable genes
  sc.pp.highly_variable_genes(data , min_mean=.125, max_mean=3, min_disp=0.25)
  data = data[:, data.var.highly_variable]
  print("The final shape of the data is {}".format(data.shape))  

#Save results
data.write(results)
print("data shape", data)
np.savetxt("labels_{}.csv".format(args.path), labels, delimiter=",")
