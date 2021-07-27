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
with gzip.open('GSE131907_Lung_Cancer_normalized_log2TPM_matrix.txt.gz', 'rb') as f:
    file_content = f.read()
