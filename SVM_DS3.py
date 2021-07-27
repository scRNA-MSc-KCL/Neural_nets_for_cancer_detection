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

#Unzip files

#with gzip.open('GSE131907_Lung_Cancer_raw_UMI_matrix.txt.gz', 'rb') as f_in:
#    with open('GSE131907_Lung_Cancer_raw_UMI_matrix.txt', 'wb') as f_out:
#        shutil.copyfileobj(f_in, f_out)
    
import csv

txt_file = r"GSE131907_Lung_Cancer_raw_UMI_matrix.txt"
csv_file = r"GSE131907_Lung_Cancer_raw_UMI_matrix.csv"

in_txt = csv.reader(open(txt_file, "rb"), delimiter = '\t')
out_csv = csv.writer(open(csv_file, 'wb'))

out_csv.writerows(in_txt)
