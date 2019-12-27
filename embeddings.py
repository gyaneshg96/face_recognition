import os
import glob
from mtcnn import MTCNN
import pandas as pd
import numpy as np
import cv2

from keras.models import load_model
from helpers import *
#from augment_image import augement_dataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
import time

ROOT_FOLDER = "lfw/"
FACENET_PATH = "../model/facenet_keras.h5"

model = load_model(FACENET_PATH)
print('Loaded Model')

dataset = []
detector = MTCNN()

i=0
start = time.time()
for path in glob.iglob(os.path.join(ROOT_FOLDER, "**", "*.jpg")):
	person = path.split("/")[-2]
	image = get_embeddings(detect_face(path,detector),model)
	dataset.append({"person":person, "image": image})
	i=i+1
	if i == 200:
		break
	if i % 20:
		print(time.time() - start)

print(time.time() - start)
dataset = pd.DataFrame(dataset)
dataset.to_pickle('images.pkl')













