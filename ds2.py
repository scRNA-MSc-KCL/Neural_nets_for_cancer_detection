import pandas as pd
import scanpy as sc

data = sc.read_csv("human_cell_atlas/krasnow_hlca_facs_counts.csv")
data_10x = sc.read_csv("human_cell_atlas/krasnow_hlca_10x_UMIs.csv")
labels = pd.read_csv("human_cell_atlas/krasnow_hlca_facs_metadata.csv")
labels_10x = pd.read_csv("human_cell_atlas/krasnow_hlca_10x_metadata.csv")


print(data)
print(data_10x)
print(labels)
print(labels_10x)
