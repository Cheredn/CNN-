# Berry Classifier — Recognizing Berries with CNN

This is a project to create a model based on a convolutional neural network (CNN) using pre-trained MobileNetV2. The model is designed to recognize different types of berries from images. Multi-class classification is supported, and the project is easily adapted to any category.

# Technologies used:

Python 3
TensorFlow / Keras
MobileNetV2
OpenCV, PIL, Matplotlib

# Project structure:

The dataset folder contains images of berries, arranged in subfolders with class names.
The classic.py file — training the model.
The classifier_use.py file — using the model for recognition.
The image_classifierV2.h5 file — the saved trained model.
The README.md file — project description.

# How to use:

** Model training **

Make sure the dataset folder contains folders with images for each class.
Run classic.py - after training, the model will be saved as image_classifierV2.h5.

** Prediction **

Make sure the image_classifierV2.h5 file exists.
Open classifier_use.py.
Inside it, the predict_image function is called with the path to the image, for example:
predict_image('arbyz.jpg')
Once run, the model will output the predicted class and display the image with the caption.

** Example output: **

Model identified: watermelon

# Notes:

- All images are automatically scaled to 128x128

- The model supports both binary and multi-class classification

- Using MobileNetV2 allows achieving good results even on a small dataset

# Example of dataset folder structure:

dataset
-  raspberry
 img1.jpg
 img2.jpg
-  strawberry
 img1.jpg
 img2.jpg
-  watermelon
 img1.jpg
 img2.jpg

# License:

The project is open and freely available for use. You can modify and use the code as you wish. Attribution is welcome, but not required.
