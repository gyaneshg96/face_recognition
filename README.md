# Face Recognition
Face Recognition and Detection using Pretrained FaceNet

We are using pretrained weights for running face recognition algorithm
The basic pipeline for prediction, given 2 images is:

* Detect the bounding box for face using MTCNN and crop it
* Normalize the image and resize to 160*160*3
* Pass throught the pretrained FaceNet model
* Pass the resultant 128-size vector to any classifier (Kmeans or SVM for time being)

The same class denotes same person and vice versa

## Contents
The above repo also contains script for augmenting image. This ensures that there are 10 examples per class.
This will be used in the fine-tuning task for FaceNet, which is to be done.
The model files in the repo can be reused and are created after sufficient GPU usage.
