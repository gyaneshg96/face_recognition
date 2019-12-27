import numpy as np
from mtcnn import MTCNN
import cv2
from numpy import expand_dims
from PIL import Image
from sklearn.preprocessing import Normalizer, LabelEncoder


def detect_face(filename, detector):
	img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
	results = detector.detect_faces(img)
	x1, y1, width, height = results[0]['box']
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
	face = img[y1:y2, x1:x2]
	return face


def get_embeddings(face_pixels, model):
	#center the pixel
	new_width  = 160
	new_height = 160
	face_pixels = np.array(Image.fromarray(face_pixels).resize((new_width, new_height), Image.ANTIALIAS))
	face_pixels = face_pixels.astype('float32')
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	samples = expand_dims(face_pixels, axis=0)
	samples.shape
	# make prediction to get embedding
	yhat = model.predict(samples)
	return yhat[0]

def normalizer(trainX, testX):
	in_encoder = Normalizer(norm='l2')
	trainX = in_encoder.transform(trainX)
	testX = in_encoder.transform(testX)
	return trainX, testX

def normalizer(trainX):
	in_encoder = Normalizer(norm='l2')
	trainX = in_encoder.transform(trainX)
	return trainX

def labeller(trainy):
	out_encoder = LabelEncoder()
	out_encoder.fit(trainy)
	trainy = out_encoder.transform(trainy)
	return trainy

def labeller(trainy, testy):
	out_encoder = LabelEncoder()
	out_encoder.fit(trainy)
	trainy = out_encoder.transform(trainy)
	testy = out_encoder.transform(testy)
	return trainy, testy
