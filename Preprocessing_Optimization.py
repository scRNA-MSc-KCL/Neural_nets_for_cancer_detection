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
import sklearn

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

def create_figures(data, filter_method, filter_by_highly_variable_genes):
    sc.pl.highest_expr_genes(data, n_top=20, save ='highly_expressed_genes.png')
    data.var['mt'] = data.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(data, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    sc.pl.violin(data, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
             jitter=0.4, multi_panel=True, save = 'mitochonrial_and_violin_plots.png')
    sc.pl.scatter(data, x='total_counts', y='pct_counts_mt', save = 'mitochonrial_and_violin_plots.png')
    sc.pl.scatter(data, x='total_counts', y='n_genes_by_counts', save ='pct_counts_mt_scatter.png')
    if filter_method == "highly_variable":
      sc.pp.highly_variable_genes(data, n_top_genes=filter_by_highly_variable_genes)
      data = data[:, data.var.highly_variable]
      sc.pl.highly_variable_genes(data, save = 'highly_variable_summary_stats.png')
    else:
      sc.pp.highly_variable_genes(data, min_mean=0.0125, max_mean=3, min_disp=0.5)
      data = data[:, data.var.highly_variable]
      sc.pl.highly_variable_genes(data, save = 'highly_variable_summary_stats.png')

# filter based on summary statistics

def SVM_Optimizer_Method_1(data, labels, filter_genes, min_mean, max_mean, mean_disp, normalize, unit_var):
  filter_genes_list = []
  normalize_list = []
  unit_var_list = []
  percentage_accuracy_list = []
  filter_method = []
  data_shape_list = []
  max_mean_list =[]
  min_mean_list = []
  mean_disp_list = []
  print("origional shape {}".format(data.shape))
  #filter data 
  for a in filter_genes:
    filtered_1_data = data.copy()
    sc.pp.filter_genes(filtered_1_data, min_cells=a)
    #normalize data
    for b in normalize:
      normalized_data = filtered_1_data.copy()
      if b == "yes":
        sc.pp.normalize_total(normalized_data, target_sum=10000)
        #logarithmize data
      logarithmized_data = normalized_data.copy()
      sc.pp.log1p(logarithmized_data)
      #filter genes
      for c in min_mean:
        for d in max_mean:
          for f in mean_disp:
            filtered_2_data = logarithmized_data.copy()
            sc.pp.highly_variable_genes(filtered_2_data, min_mean=c, max_mean=d, min_disp=f)
            filtered_2_data = filtered_2_data[:, filtered_2_data.var.highly_variable]
            for e in unit_var:
              filtered_3_data = filtered_2_data.copy()
              if e == "yes":
                sc.pp.scale(filtered_3_data, max_value=10)
              print("final data shape", filtered_3_data.shape)
              filter_genes_list.append(a)
              normalize_list.append(b)
              unit_var_list.append(e)
              max_mean_list.append(c)
              min_mean_list.append(d)
              mean_disp_list.append(f)
              filter_method.append("filter_based_on_stats")
              X_train, X_test, y_train, y_test = train_test_split(filtered_3_data.X, labels, test_size=0.2, random_state=42)
              Classifier = sklearn.svm.SVC(kernel = "rbf")
              Classifier.fit(X_train, y_train)
              print("the classification result with the current settings and a {} kernal is {}".format("rbf", Classifier.score(X_test, y_test)))
              percentage_accuracy = (Classifier.score(X_test, y_test))*100 
              percentage_accuracy_list.append(percentage_accuracy)  
              data_shape_list.append(filtered_3_data.shape)
              df = pd.DataFrame(list(zip(filter_genes_list, normalize_list, filter_method, max_mean_list, min_mean_list, mean_disp_list, unit_var_list, percentage_accuracy_list, data_shape_list)),
                        columns =['Min_number_of_cells_per_gene', 'normalized', "filter_method", "max_mean", "min_mean", "mean_disp", "scaled_to_unit_var", "percentage_accuracy", "data_shape"])
  return df
   
#filter based on variable genes

def SVM_Optimizer_Method_2(data, labels, filter_genes, normalize, filter_by_highly_variable_gene, unit_var):
  filter_genes_list = []
  normalize_list = []
  filter_by_highly_variable_genes_list = []
  unit_var_list = []
  percentage_accuracy_list = []
  filter_method = []
  data_shape_list = []
  #filter data 
  for a in filter_genes:
    filtered_1_data = data.copy()
    sc.pp.filter_genes(filtered_1_data, min_cells=a)
    print("after filtering genes with min cells", filtered_1_data.shape)
    #normalize data
    for b in normalize:
      normalized_data = filtered_1_data.copy()
      if b == "yes":
        sc.pp.normalize_total(normalized_data, target_sum=10000)
        #logarithmize data
      logarithmized_data = normalized_data.copy()
      sc.pp.log1p(logarithmized_data)
      #filter genes
      for d in filter_by_highly_variable_gene:
        filtered_2_data = logarithmized_data.copy()
        sc.pp.highly_variable_genes(filtered_2_data, n_top_genes=d)
        filtered_2_data = filtered_2_data[:, filtered_2_data.var.highly_variable]
        for e in unit_var:
          filtered_3_data = filtered_2_data.copy()
          if e == "yes":
            sc.pp.scale(filtered_3_data, max_value=10)
            print("final data shape", filtered_3_data.shape)
          else:
            print("not clipped", filtered_3_data.shape)
          filter_genes_list.append(a)
          normalize_list.append(b)
          filter_by_highly_variable_genes_list.append(d)
          unit_var_list.append(e)
          filter_method.append("filter_based_on_variable_genes")
          X_train, X_test, y_train, y_test = train_test_split(filtered_3_data.X, labels, test_size=0.2, random_state=42)
          Classifier = sklearn.svm.SVC(kernel = "rbf")
          Classifier.fit(X_train, y_train)
          print("the classification result with the current settings and a {} kernal is {}".format("rbf", Classifier.score(X_test, y_test)))
          percentage_accuracy = (Classifier.score(X_test, y_test))*100 
          percentage_accuracy_list.append(percentage_accuracy) 
          data_shape_list.append(filtered_3_data.shape)
  df = pd.DataFrame(list(zip(filter_genes_list, normalize_list, filter_method, filter_by_highly_variable_genes_list, unit_var_list, percentage_accuracy_list, data_shape_list)),
                        columns =['Min_number_of_cells_per_gene', 'normalized', "filter_method", "number_of_top_genes", "scaled_to_unit_var", "percentage_accuracy", "data_shape"])
  return df      

start = time.time()

parser = argparse.ArgumentParser(description='Select dataset')
parser.add_argument('path', type = int)
args = parser.parse_args()
if args.path == 1:
  labels =pd.read_csv("Original_data/Labels.csv")
  data = sc.read_csv("Original_data/Combined_10x_CelSeq2_5cl_data.csv")
  file_loc = "test_results/DS1/SVM"
if args.path == 2:
  data = sc.read_csv("Original_data/human_cell_atlas/krasnow_hlca_10x_UMIs.csv") #26485 x 65662
  data = anndata.AnnData.transpose(data)
  #labels = pd.read_csv("human_cell_atlas/krasnow_hlca_facs_metadata.csv") #9409 x 141
  ##data = sc.read_csv("human_cell_atlas/krasnow_hlca_facs_counts.csv")  #58683 x 9409
  labels = pd.read_csv("Original_data/human_cell_atlas/krasnow_hlca_10x_metadata.csv") #65662 x 21
  labels = labels["free_annotation"]
  file_loc = "test_results/DS2/SVM"
if args.path == 3:
  labels =pd.read_csv("Original_data/GSE131907_Lung_Cancer_cell_annotation.txt", sep = "\t")
  data = sc.read_csv("Original_data/GSE131907_Lung_Cancer_raw_UMI_matrix.csv") #29634 x 208506
  data = anndata.AnnData.transpose(data)
  labels = labels["Cell_type"]
  file_loc = "test_results/DS3/SVM"
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
  file_loc = "test_results/DS4/SVM"


#make directory for results
path = os.getcwd()
path = os.path.join(path, "{}/{}".format(file_loc,start))
try:
  os.makedirs(path)
except OSError:
  print("Creation of the directory %s failed" % path)
else:
  print("Successfully created the directory %s" % path)

labels = label_adaption(labels)

filter_genes = [1, 5]
#filter_genes = [1]
normalize = ["yes", "no"]
#normalize = ["no"]
filter_by_highly_variable_gene = [500, 1000, 2000]
#filter_by_highly_variable_gene = [500]
min_mean = [.125, .150]
max_mean = [3, 6]
mean_disp = [.5, 1]
#min_mean = [0.125]
#max_mean = [3]
#mean_disp = [0.5]
unit_var= ["yes", "no"]

            
results_dataframe_method_1 = SVM_Optimizer_Method_1(data, labels, filter_genes,  min_mean, max_mean, mean_disp, normalize, unit_var)  
results_dataframe_method_1.to_csv("{}/{}/Method_1.csv".format(file_loc, start))  

#results_dataframe_method_2 = SVM_Optimizer_Method_2(data, labels, filter_genes, normalize, filter_by_highly_variable_gene, unit_var)  
#results_dataframe_method_2.to_csv("{}/{}/Method_2.csv".format(file_loc, start))  

end = time.time()
print("The time taken to complete this program was {}".format(end - start))
