from image_bound import detect_face
from helpers import get_embeddings

filename1 = sys.argv[1]
filename2 = sys.argv[2]

#pipeline

FACENET_PATH = "models/facenet_keras.h5"
KMEANS_PATH="models/kmeans_model.sav"
SVM_PATH="models/svm_model.sav"
#FACENET_TRAINED

embedding1 = get_embeddings(detect_face(filename1))
embedding2 = get_embeddings(detect_face(filename2))

model = sys.argv[3]
if model == 'kmeans':
	loaded_model = pickle.load(open(KMEANS_PATH, 'rb'))
elif model == 'svm':
	loaded_model = pickle.load(open(SVM_PATH, 'rb'))
else:
	print("No such model")