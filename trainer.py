import os
import glob
import pandas as pd
import numpy as np
import pickle

from sklearn.cluster import KMeans
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from helpers import *
import time

KMEANS_FILE="models/kmeans_model.sav"
SVM_FILE="models/svm_model.sav"
IMAGE_EMBEDDINGS="models/images.pkl"

dataset = pd.read_pickle(IMAGE_EMBEDDINGS)


datasetX = np.array(dataset['image'].to_list())
datasety = np.array(dataset['person'].to_list())

datasetX = normalizer(datasetX)
datasety = labeller(datasety)

trainX, testX, trainy, testy = train_test_split(datasetX, datasety, test_size=0.25)

nclasses = len(set(trainy))

kmeans = KMeans(nclasses, verbose=1)
svm = SVC(gamma='auto')

start = time.time()
kmeans.fit(trainX)
start = time.time()

train_kmeans_result = kmeans.predict(trainX)

start = time.time()
svm.fit(trainX, trainy)
start = time.time()

train_svm_result = svm.predict(trainX)

kmeans.predict(testX)
svm.predict(testX)

pickle.dump(kmeans, open(KMEANS_FILE, 'wb'))
pickle.dump(svm, open(SVM_FILE, 'wb'))
















