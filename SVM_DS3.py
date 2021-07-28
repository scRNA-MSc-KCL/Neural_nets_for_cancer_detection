import pandas as pd
import scanpy as sc
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import anndata
import zipfile
import gzip
import shutil

import os
filename = 'GSE131907_Lung_Cancer_raw_UMI_matrix.txt.gz'
os.system('gunzip ' + filename)
#Unzip files

import csv

txt_file = "GSE131907_Lung_Cancer_raw_UMI_matrix.txt"
csv_file = "GSE131907_Lung_Cancer_raw_UMI_matrix.csv"

in_txt = csv.reader(open(txt_file, "r"), delimiter = '\t')
out_csv = csv.writer(open(csv_file, 'w'))
out_csv.writerows(in_txt)

l = sc.read_csv("GSE131907_Lung_Cancer_raw_UMI_matrix.csv")
print(l)
