import os
import glob
from mtcnn import MTCNN
import pandas as pd
import numpy as np
import cv2

from keras.models import load_model
from helpers import *

import time


from keras.preprocessing.image import ImageDataGenerator 
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    horizontal_flip=True)

ROOT_FOLDER = "lfw/"


#max images per person

MAX_IMG = 10
start = time.time()
dataset2 = []
j = 0
for path in glob.iglob(os.path.join(ROOT_FOLDER, "**")):
  i = 0
  j = j + 1
  print(j)
  personname=path.split("/")[-1]
  person = []
  for filename in glob.iglob(path+'/*.jpg'):
    i = i+1
    img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    person.append(img)
    if i == MAX_IMG:
      break
  person = np.asarray(person)
  gen = datagen.flow(person, batch_size=1)
  for i in range(MAX_IMG):
    batch = gen.next()
    dataset2.append({"person":person, "image": batch[0]})
  del person
  del img

dataset2 = pd.DataFrame(dataset2)