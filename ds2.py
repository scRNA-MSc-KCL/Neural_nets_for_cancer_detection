import pandas as pd
import scanpy as sc
import numpy as np
import sklearn
import tensorflow
import keras
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import anndata

data = sc.read_csv("human_cell_atlas/krasnow_hlca_facs_counts.csv")
data_10x = sc.read_csv("human_cell_atlas/krasnow_hlca_10x_UMIs.csv")
labels = pd.read_csv("human_cell_atlas/krasnow_hlca_facs_metadata.csv")
labels_10x = pd.read_csv("human_cell_atlas/krasnow_hlca_10x_metadata.csv")


print("facs counts", data)
print("UMI counts", data_10x)
print("facs labels", labels)
print("10x labels", labels_10x)


