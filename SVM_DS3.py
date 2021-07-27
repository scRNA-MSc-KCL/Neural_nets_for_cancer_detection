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

f_in = open('GSE131907_Lung_Cancer_raw_UMI_matrix.txt')
f_out = gzip.open('GSE131907_Lung_Cancer_raw_UMI_matrix.txt.gz', 'wb')
f_out.writelines(f_in)
f_out.close()
f_in.close()

    
#import csv

#txt_file = r"mytxt.txt"
#csv_file = r"mycsv.csv"

#in_txt = csv.reader(open(txt_file, "rb"), delimiter = '\t')
#out_csv = csv.writer(open(csv_file, 'wb'))

#out_csv.writerows(in_txt)
