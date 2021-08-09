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
import time
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

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

start = time.time()

#3 is UMI raw
#4 is normalized

parser = argparse.ArgumentParser(description='Select dataset')
parser.add_argument('path', type = int)

args = parser.parse_args()
if args.path == 1:
  #Unzip files - for dataset 1
#with zipfile.ZipFile("Dataset1_interdataset.zip", 'r') as zip_ref:
#    zip_ref.extractall()
  labels =pd.read_csv("Original_data/Labels.csv")
  data = sc.read_csv("Original_data/Combined_10x_CelSeq2_5cl_data.csv")
  file_loc = "test_results/DS1/SVM"

labels = label_adaption(labels)

filter_genes = [1, 5, 10]
remove_high_counts = [1000, 2500, 5000]
normalize = ["yes", "no"]
filter_method = ["highly_variable", "summary_stat"]
filter_by_highly_variable_genes = [500, 1000, 2000]
min_mean = 0.125
max_mean = 3
mean_disp = 0.5
regress_data = ["yes", "no"]
unit_variance = ["yes", "no"]
FIGS = "n"

#def SVM_Optimizer(filter_genes, remove_high_counts, normalize, filter_method, filter_by_highly_variable_genes, regress_data, unit
if FIGS == "y":
  sc.pl.highest_expr_genes(data, n_top=20, save ='highly_expressed_genes.png')

#read data
print("The original shape of the data1 is {}".format(data))

#filter data 
sc.pp.filter_genes(data, min_cells=3)
print("A", data.shape)
#remove mitochonrial
adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
if FIGS == "y":
  data.var['mt'] = data.var_names.str.startswith('MT-')
  sc.pp.calculate_qc_metrics(data, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
  sc.pl.violin(data, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
             jitter=0.4, multi_panel=True, save = 'mitochonrial_and_violin_plots.png')
  sc.pl.scatter(data, x='total_counts', y='pct_counts_mt', save = 'mitochonrial_and_violin_plots.png')
  sc.pl.scatter(data, x='total_counts', y='n_genes_by_counts', save ='pct_counts_mt_scatter.png')
#data = data[data.obs.n_genes_by_counts < 2500, :]
#data = data[data.obs.pct_counts_mt < 5, :]
#normalize data
sc.pp.normalize_total(data, target_sum=10000)
print("C", data.shape)
#logarithmize data
sc.pp.log1p(data)
print("D", data.shape)

#select top x highly variable genes
#sc.pp.highly_variable_genes(data, n_top_genes=1000)
#data = data[:, data.var.highly_variable]

#select highly variable genes based on summary statistics
sc.pp.highly_variable_genes(data, min_mean=0.0125, max_mean=3, min_disp=0.5)
if FIGS == "y":
  sc.pl.highly_variable_genes(data, save = 'highly_variable_summary_stats.png')
data = data[:, data.var.highly_variable]
print("E", data.shape)

#regress out data
sc.pp.regress_out(data, ['total_counts', 'pct_counts_mt'])
print("F", data.shape)

#scale to unit variance
sc.pp.scale(data, max_value=10)
print("G", data.shape)

print("The final shape of the data is {}".format(data.shape))

#create training and test sets
X_train, X_test, y_train, y_test = train_test_split(data.X, labels, test_size=0.2, random_state=42)

SVM_list = ["linear"]

# build classifier
                                                    
for i in SVM_list:
  Classifier = sklearn.svm.SVC(kernel = i)
  Classifier.fit(X_train, y_train)
  print("the classification result with the current settings and a {} kernal is {}".format(i, Classifier.score(X_test, y_test)))
  
  
end = time.time()
print("The time taken to complete this program was {}".format(end - start))
