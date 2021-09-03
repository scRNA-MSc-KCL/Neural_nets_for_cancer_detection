from pyDeepInsight import ImageTransformer, LogScaler
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D 
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import BatchNormalization
import numpy as np
import pandas as pd
import sklearn
import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import requests
import zipfile
import csv
import scanpy as sc
import argparse
import anndata
import time
import os

##Note; The transformation of convolutional data to image data was transformed using DeepInsight (ref;https://www.nature.com/articles/s41598-019-47765-6)

start = time.time()

#Load data
parser = argparse.ArgumentParser(description='Select dataset')
parser.add_argument('path', type = int)
args = parser.parse_args()
#Dataset 1
if args.path == 1:
  labels =pd.read_csv("labels_1.csv", names = ["X"])
  data = sc.read("results_1.h5ad")
  file_loc = "test_results/DS1/CNN"
  b =50
#Dataset 2
if args.path == 2:
  labels =pd.read_csv("labels_2.csv", names = ["X"])
  data = sc.read("results_2.h5ad")
  file_loc = "test_results/DS2/CNN"
  b = 500
#Dataset 3
if args.path == 4:
  labels =pd.read_csv("labels_4.csv", names = ["X"])
  data = sc.read("results_4.h5ad")
  file_loc = "test_results/DS3/CNN"
  b = 50
  
#Establish Variables
num_lab = len(labels["X"].unique())
counter = 0
e = 50


#Create new folder for results
path = os.getcwd()
path = os.path.join(path, "{}/{}".format(file_loc,start))
try:
  os.makedirs(path)
except OSError:
  print("Creation of the directory %s failed" % path)
else:
  print("Successfully created the directory %s" % path)

#perform pca on data
print("The original shape of the data is {}".format(data.shape))
sc.tl.pca(data, svd_solver='arpack')

#split data
X_train, X_test, y_train, y_test = train_test_split(data.X, labels, test_size=0.33, random_state=42)
print("train test split performed {}".format(time.time() - start))

#make labels for neural network catagorical
y_train = to_categorical(y_train, num_lab)
y_test = to_categorical(y_test, num_lab)
X_train_norm = X_train
X_test_norm = X_test

#split data using pca
it = ImageTransformer(feature_extractor='pca', 
                      pixels=50, random_state=1701, 
                      n_jobs=-1)
fig = plt.figure(figsize=(5, 5))
_ = it.fit(X_train_norm, plot=True)

#save figure created
fig.savefig('{}/{}/fig_1'.format(file_loc, start))
print("fit transform performed{}".format(time.time() - start))
#convert to pixel image version
fdm = it.feature_density_matrix()
fdm[fdm == 0] = np.nan
fig = plt.figure(figsize=(10, 7))

#Create heat map
ax = sns.heatmap(fdm, cmap="viridis", linewidths=0.01, 
                 linecolor="lightgrey", square=True)
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
for _, spine in ax.spines.items():
    spine.set_visible(True)
_ = plt.title("Genes per pixel")
fig.savefig('{}/{}/fig_2'.format(file_loc, start))

#Image transformation
X_train_img = it.transform(X_train_norm)
X_train_img = it.fit_transform(X_train_norm)
X_test_img = it.transform(X_test_norm)
X_train_img = X_train_img.reshape(X_train_img.shape[0], 50, 50, 3)
X_test_img = X_test_img.reshape(X_test_img.shape[0], 50, 50, 3)
print("image transformation completed {}".format(time.time() - start))

#Build CNN
net = Sequential()
net.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu',
input_shape=(50,50,3)))
net.add(BatchNormalization())
#net.add(Conv2D(64, (3, 3), activation='relu'))
net.add(MaxPool2D(pool_size=(2, 2)))
net.add(Flatten())
net.add(Dense(256, activation='relu'))
net.add(Dropout(rate=0.5))
net.add(Dense(num_lab, activation='softmax'))

net.summary()
from contextlib import redirect_stdout

#print model summary
with open('{}/{}/model_summary.txt'.format(file_loc, start), 'w') as f:
    with redirect_stdout(f):
        net.summary()

#train CNN
print("start training {}".format(time.time() - start))
net.compile(loss='categorical_crossentropy', optimizer='adam')
history = net.fit(X_train_img, y_train,
validation_data=(X_test_img, y_test),
 epochs=e,
 batch_size=b)
print("end training {}".format(time.time() - start))

#get CNN plot
fig = plt.figure()
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
fig.savefig('{}/{}/fig_3'.format(file_loc, start))

#Calculate Network accuracy
outputs = net.predict(X_test_img)
print("outputs predicted {}".format(time.time() - start))
labels_predicted= np.argmax(outputs, axis=1)
y_test_decoded = np.argmax(y_test, axis=1)  # maybe change so you're not doing every time
misclassified =  (np.sum(labels_predicted != y_test_decoded)/(len(y_test_decoded)))*100
f = open('{}/{}/model_summary.txt'.format(file_loc, start), 'a')
f.write("percentage missclassified on test set is {}\n".format(misclassified))
f.write("number of epochs is {}".format(e))
print("misclassified; ", misclassified)
end = time.time()
f.write("The time taken to complete this program was {}".format(end - start))
print("The time taken to complete this program was {}".format(end - start))
f.close()
