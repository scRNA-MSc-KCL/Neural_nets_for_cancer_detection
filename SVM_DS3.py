import pandas as pd
import scanpy as sc
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import anndata
import zipfile


#Unzip files
import gzip
with gzip.open('GSE131907_Lung_Cancer_raw_UMI_matrix.txt.gz', 'rb') as f:
    read_file.to_csv(f)

    
l = sc.read_csv("GSE131907_Lung_Cancer_raw_UMI_matrix.csv")
print(l)
 
