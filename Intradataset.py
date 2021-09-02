import pandas as pd
import scanpy as sc
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time
import argparse
import anndata
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical

labels_10x =pd.read_csv("labels_5.csv", names = ["X"])
labels_10x = labels_10x['X']
data_10x = sc.read("results_5.h5ad")
labels_celseq =pd.read_csv("labels_6.csv", names = ["X"])
labels_celseq = labels_celseq['X']
print(len(labels_celseq))
data_celseq = sc.read("results_6.h5ad")

#K Nearest Neighbours

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(data_10x.X, labels_10x)
print("The intradataset nearest neighbour score;", neigh.score(data_celseq.X, labels_celseq))

#SVM

Classifier = sklearn.svm.SVC(kernel = 'rbf')
Classifier.fit(data_10x.X, labels_10x)
print("The intradataset SVM score {}".format(Classifier.score(data_celseq.X, labels_celseq)))
                                                              
#Multilayer Perceptron
X_train, X_val, y_train, y_val = train_test_split(data_10x.X, labels_10x, test_size=0.2, random_state=42)
#make labels for neural network catagorical
y_train = to_categorical(y_train, 5)
y_test = to_categorical(labels_celseq, 5)
y_val = to_categorical(y_val, 5)

#Build Network
net = Sequential()
net.add(Dense(500, activation = 'relu', input_shape = (data_10x.n_vars,)))
net.add(Dense(5, activation='softmax'))
net.compile(loss="categorical_crossentropy", optimizer='Adam')
history = net.fit(X_train, y_train,validation_data=(X_val, y_val),epochs=7,batch_size=50)
outputs = net.predict(data_celseq.X)
labels_predicted= np.argmax(outputs, axis=1)
y_test_decoded = np.argmax(y_test, axis=1)  # maybe change so you're not doing every time
correctly_classified =  (np.sum(labels_predicted == y_test_decoded)/(len(y_test_decoded)))*100
print("The intradataset MLP score {}".format(correctly_classified))


#Convolutional Neural Network
it = ImageTransformer(feature_extractor='pca', 
                      pixels=50, random_state=1701, 
                      n_jobs=-1)
fig = plt.figure(figsize=(5, 5))
_ = it.fit(X_train, plot=True)
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
                                                              
X_train_img = it.transform(X_train)
X_train_img = it.fit_transform(X_train)
X_test_img = it.transform(X_test)
X_train_img = X_train_img.reshape(X_train_img.shape[0], 50, 50, 3)
X_test_img = X_test_img.reshape(X_test_img.shape[0],50, 50, 3)

net = Sequential()
net.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu',
input_shape=(50,50,3)))
net.add(BatchNormalization())
net.add(MaxPool2D(pool_size=(2, 2)))
net.add(Flatten())
net.add(Dense(256, activation='softplus'))
net.add(Dropout(rate=0.2))           
net.add(Dense(num_lab, activation='softmax'))
net.summary()
net.compile(loss='categorical_crossentropy', optimizer='adamax')
history = net.fit(X_train_img, y_train,validation_data=(X_test_img, y_test),
epochs=50,
batch_size=50)
outputs = net.predict(X_test_img)
labels_predicted= np.argmax(outputs, axis=1)
y_test_decoded = np.argmax(y_test, axis=1)  # maybe change so you're not doing every time
accuracy =  (np.sum(labels_predicted == y_test_decoded)/(len(y_test_decoded)))*100
print("The intradataset CNN score {}".format(correctly_classified))
