import pandas as pd
import scanpy as sc
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import anndata


#Unzip files
with zipfile.ZipFile("index.html?acc=GSE131907", 'r') as zip_ref:
    zip_ref.extractall()
