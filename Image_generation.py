  
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
  data = sc.read_csv("Original_data/Combined_10x_CelSeq2_5cl_data.csv")
  file_loc = "DS1/Fig"
if args.path == 2:
  data = sc.read_csv("Original_data/human_cell_atlas/krasnow_hlca_10x_UMIs.csv") #26485 x 65662
  data = anndata.AnnData.transpose(data)
  #labels = pd.read_csv("human_cell_atlas/krasnow_hlca_facs_metadata.csv") #9409 x 141
  ##data = sc.read_csv("human_cell_atlas/krasnow_hlca_facs_counts.csv")  #58683 x 9409
  file_loc = "DS2/Fig"
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
  file_loc = "DS4/Fig"
  
  
path = os.getcwd()
path = os.path.join(path, "test_results/{}".format(file_loc))
try:
  os.makedirs(path)
except OSError:
  print("Creation of the directory %s failed" % path)
else:
  print("Successfully created the directory %s" % path)

#Pipeline 1
if args.path == 1 or args.path == 4: 
  #print highly expressed genes
  fig = sc.pl.highest_expr_genes(data, n_top=20, )
  fig.savefig('test_results/{}/highly_expressed}'.format(file_loc))
  #filtering
  sc.pp.filter_genes(data, min_cells=1)
  #normalize data
  sc.pp.normalize_total(data, target_sum=10000)
  #logarithmize data
  sc.pp.log1p(data)
  #select highly variable genes
  sc.pp.highly_variable_genes(data, n_top_genes=2000)
  data = data[:, data.var.highly_variable]
  print("The final shape of the data is {}".format(data.shape))

#Pipeline 2
if args.path == 2 or args.path == 3: 
  #print highly expressed genes
  sc.pl.highest_expr_genes(data, n_top=20, )
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

