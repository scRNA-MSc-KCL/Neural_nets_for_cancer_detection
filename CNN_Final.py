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
from sklearn.model_selection import KFold
from pycm import *
import seaborn
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from numpy import savetxt
accuracy_list = []
run_time_list = []
#Load data
start = time.time()

parser = argparse.ArgumentParser(description='Select dataset')
parser.add_argument('path', type = int)
args = parser.parse_args()
if args.path == 1:
  labels =pd.read_csv("labels_1.csv", names = ["X"])
  data = sc.read("results_1.h5ad")
  file_loc = "test_results/DS1/CNN/Final"
  b = 50
if args.path == 2:
  labels =pd.read_csv("labels_2.csv", names = ["X"])
  data = sc.read("results_2.h5ad")
  file_loc = "test_results/DS2/CNN/Final"
  b = 500
if args.path == 4:
  labels =pd.read_csv("labels_4.csv", names = ["X"])
  data = sc.read("results_4.h5ad")
  file_loc = "test_results/DS4/CNN/Final"
  b = 50
num_lab = len(labels["X"].unique())
counter = 0

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

X_split, X_test, y_split, y_test = train_test_split(data.X, labels, test_size=0.2)
y_split = y_split.reset_index(drop = True)
y_test = y_test.reset_index(drop = True)
y_test = to_categorical(y_test, num_lab)

kf = KFold(n_splits=5, shuffle = True)
for train_index, test_index in kf.split(X_split):
  X_train, X_val = X_split[train_index], X_split[test_index]
  y_train, y_val = y_split['X'][train_index], y_split['X'][test_index]
  y_train = to_categorical(y_train, num_lab)
  y_val = to_categorical(y_val, num_lab)
  X_train_norm = X_train
  X_test_norm = X_test
  if args.path == 1 or args.path == 4:
    p = 50
    fe = 'pca'
  if args.path == 2:
    p = 100
    fe = 'kpca'
  it = ImageTransformer(feature_extractor=fe, 
                      pixels=p, random_state=1701, 
                      n_jobs=-1)
  fig = plt.figure(figsize=(5, 5))
  _ = it.fit(X_train_norm, plot=True)

  fig.savefig('{}/{}/fig_1'.format(file_loc, start))

  #convert to pixel image version
  fdm = it.feature_density_matrix()
  fdm[fdm == 0] = np.nan
  fig = plt.figure(figsize=(10, 7))

  ax = sns.heatmap(fdm, cmap="viridis", linewidths=0.01, 
                   linecolor="lightgrey", square=True)
  ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
  for _, spine in ax.spines.items():
      spine.set_visible(True)
  _ = plt.title("Genes per pixel")

  fig.savefig('{}/{}/fig_2'.format(file_loc, start))

  X_train_img = it.transform(X_train_norm)
  X_train_img = it.fit_transform(X_train_norm)
  X_test_img = it.transform(X_test_norm)

  X_train_img = X_train_img.reshape(X_train_img.shape[0], p, p, 3)
  X_test_img = X_test_img.reshape(X_test_img.shape[0], p, p, 3)

  
#Build CNN
  net = Sequential()
  net.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu',
  input_shape=(p,p,3)))
  net.add(BatchNormalization())
  net.add(MaxPool2D(pool_size=(2, 2)))
  net.add(Flatten())
  if args.path == 1:
    net.add(Dense(256, activation='softplus'))
    net.add(Dropout(rate=0.2))
  if args.path == 2:
    net.add(Dense(400, activation='softplus'))
    net.add(Dropout(rate=0.2))
  if args.path ==4:
    net.add(Dense(200, activation='softplus'))
    net.add(Dropout(rate=0.2))              
  net.add(Dense(num_lab, activation='softmax'))
  net.summary()
  from contextlib import redirect_stdout

  with open('{}/{}/model_summary{}.txt'.format(file_loc, start, counter), 'w') as f:
      with redirect_stdout(f):
          net.summary()

#train CNN
  net.compile(loss='categorical_crossentropy', optimizer='adamax')
  history = net.fit(X_train_img, y_train,
  validation_data=(X_test_img, y_test),
   epochs=50,
   batch_size=b)

#get CNN plot
  fig = plt.figure()
  plt.plot(history.history['loss'], label='training loss')
  plt.plot(history.history['val_loss'], label='validation loss')
  plt.xlabel('epochs')
  plt.ylabel('loss')
  plt.legend()
  fig.savefig('{}/{}/fig_3'.format(file_loc, start))

  outputs = net.predict(X_test_img)
  labels_predicted= np.argmax(outputs, axis=1)
  y_test_decoded = np.argmax(y_test, axis=1)  # maybe change so you're not doing every time
  accuracy =  (np.sum(labels_predicted == y_test_decoded)/(len(y_test_decoded)))*100
  accuracy_list.append(accuracy)
  f = open('{}/{}/model_summary{}.txt'.format(file_loc, start,counter), 'a')
  f.write("percentage correct on test set is {}\n".format(accuracy))
  f.write("precision score: {}".format(precision_score(y_test_decoded, labels_predicted, average=None)))
  f.write("recall score: {} ".format(recall_score(y_test_decoded, labels_predicted, average=None)))
  f.write("accuracy: {}".format(accuracy))
  print("accuracy; ", accuracy)
  print(precision_score(y_test_decoded, labels_predicted, average=None))
  print(recall_score(y_test_decoded, labels_predicted, average=None))
  savetxt("{}/{}/{}_ypred.csv".format(file_loc, start, counter), labels_predicted, delimiter=',')
  savetxt("{}/{}/{}_ytrue.csv".format(file_loc, start, counter), y_test_decoded, delimiter=',')
  counter += 1
  f.close()

end = time.time()
run_time = end - start
print("The time taken to complete this program was {}".format(end - start))
df = pd.DataFrame(list(zip(accuracy_list)),columns =['accuracy'])
df.to_csv("{}/{}.csv".format(file_loc, start))
  
