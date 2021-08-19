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

accuracy_list = []
run_time_list = []
pixel_list = []
feature_list = []
cnn_filters_list = []
cnn_layers_list = []

pixels = [50]
feature_extract = ['pca']
cnn_filter = [32, 64, 128]
cnn_layers = [1,2]

#Load data
start = time.time()

parser = argparse.ArgumentParser(description='Select dataset')
parser.add_argument('path', type = int)
args = parser.parse_args()
if args.path == 1:
  labels =pd.read_csv("labels_1.csv", names = ["X"])
  data = sc.read("results_1.h5ad")
  file_loc = "test_results/DS1/CNN"
if args.path == 2:
  labels =pd.read_csv("labels_2.csv", names = ["X"])
  data = sc.read("results_2.h5ad")
  file_loc = "test_results/DS2/CNN"
if args.path == 3:
  labels =pd.read_csv("labels_3.csv", names = ["X"])
  data = sc.read("results_3.h5ad")
  file_loc = "test_results/DS3/CNN"
if args.path == 4:
  labels =pd.read_csv("labels_4.csv", names = ["X"])
  data = sc.read("results_4.h5ad")
  file_loc = "test_results/DS4/CNN"
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

#split data
X_train, X_test, y_train, y_test = train_test_split(data.X, labels, test_size=0.2, random_state=42)
X_test, X_val, y_test, y_val= train_test_split(X_test, y_test, test_size=0.5, random_state=42)

#make labels for neural network catagorical
y_train = to_categorical(y_train, num_lab)
y_test = to_categorical(y_test, num_lab)
y_val= to_categorical(y_val, num_lab)

for p in pixels:
  for f in feature_extract:
    it = ImageTransformer(feature_extractor=f, 
                          pixels=p, random_state=1701, 
                          n_jobs=-1)
    fig = plt.figure(figsize=(5, 5))
    _ = it.fit(X_train, plot=True)

    #fig.savefig('{}/{}/fig_1'.format(file_loc, start, p, f))

#convert to pixel image version
    fdm = it.feature_density_matrix()
    fdm[fdm == 0] = np.nan
    fig = plt.figure(figsize=(10, 7))

   # ax = sns.heatmap(fdm, cmap="viridis", linewidths=0.01, 
    #                 linecolor="lightgrey", square=True)
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
   # ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
   # for _, spine in ax.spines.items():
   #     spine.set_visible(True)
   # _ = plt.title("Genes per pixel")

   # fig.savefig('{}/{}/fig_2_pixel{}feature{}'.format(file_loc, start,p, f))

    X_train_img = it.fit_transform(X_train)
    X_test_img = it.transform(X_test)
    X_val_img = it.transform(X_val)

    X_train_img = X_train_img.reshape(X_train_img.shape[0], p, p, 3)
    X_test_img = X_test_img.reshape(X_test_img.shape[0], p, p, 3)
    X_val_img = X_val_img.reshape(X_val_img.shape[0], p, p, 3)

#Build CNN
    for l in cnn_layers:
      for fl in cnn_filter:
        net = Sequential()
        net.add(Conv2D(filters=fl, kernel_size=(5,5), activation='relu',
        input_shape=(p,p,3)))
        net.add(BatchNormalization())
        if l == 2:
          net.add(Conv2D(fl, (3, 3), activation='relu'))
          net.add(BatchNormalization())
        if l == 3:
          net.add(Conv2D(fl, (3, 3), activation='relu'))
          net.add(BatchNormalization())
        net.add(MaxPool2D(pool_size=(2, 2)))
        net.add(Flatten())
        net.add(Dense(256, activation='relu'))
        net.add(Dropout(rate=0.5))
        net.add(Dense(num_lab, activation='softmax'))
        net.summary()
        from contextlib import redirect_stdout

        with open('{}/{}/model_summary_filter_{}layer_{}.txt'.format(file_loc, start, fl, l), 'w') as fr:
            with redirect_stdout(fr):
                net.summary()

    #train CNN
        net.compile(loss='categorical_crossentropy', optimizer='adam')
        history = net.fit(X_train_img, y_train,
        validation_data=(X_val_img, y_val),
         epochs=50,
         batch_size=256)

    #get CNN plot
    #fig = plt.figure()
    #plt.plot(history.history['loss'], label='training loss')
    #plt.plot(history.history['val_loss'], label='validation loss')
    #plt.xlabel('epochs')
    #plt.ylabel('loss')
    #plt.legend()
    #fig.savefig('{}/{}/fig_3'.format(file_loc, start))

        outputs = net.predict(X_test_img)
        labels_predicted= np.argmax(outputs, axis=1)
        y_test_decoded = np.argmax(y_test, axis=1)  # maybe change so you're not doing every time
        accuracy =  (np.sum(labels_predicted == y_test_decoded)/(len(y_test_decoded)))*100
        #f = open('{}/{}/model_summary.txt'.format(file_loc, start), 'a')
        #f.write("percentage missclassified on test set is {}\n".format(misclassified))
        #print("misclassified; ", misclassified)
        print("accuracy: ", accuracy)
        end = time.time()
        run_time = end - start
        run_time_list.append(run_time)
        accuracy_list.append(accuracy)
        pixel_list.append(p)
        feature_list.append(f)
        cnn_layers_list.append(l)
        cnn_filters_list.append(fl)
        #f.write("The time taken to complete this program was {}".format(end - start))
        #print("The time taken to complete this program was {}".format(end - start))
        #f.close()

df = pd.DataFrame(list(zip(accuracy_list, run_time_list, pixel_list, feature_list,
                           cnn_layers_list,cnn_filters_list)),columns =['accuracy', 'run_time', 
                               'pixels', 'features', "cnn_layers", "cnn_filters"])
df.to_csv("{}/{}.csv".format(file_loc, start))
