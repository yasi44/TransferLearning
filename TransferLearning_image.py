import numpy as np
import cv2

import PIL.Image as Image
import os
import pathlib
import matplotlib.pylab as plt

import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# this is the size that we resize all images into
IMAGE_SHAPE = (224, 224)
imageShape_with_3_channel = IMAGE_SHAPE+(3,)

classifier = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4", input_shape=imageShape_with_3_channel)
])



# list of all labels in the pretrained  model
# tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
image_labels = []
with open("ImageNetLabels.txt", "r") as f:
    image_labels = f.read().splitlines()
print(image_labels[:5])

# test an image
hen = Image.open("hen.jpg").resize(IMAGE_SHAPE)
# before training and classification, the images must be scaled [0-255]
# every image / 255.0
hen = np.array(hen)/255.0
print(hen.shape)
#to check values are scaled
hen[np.newaxis, ...]

result = classifier.predict(hen[np.newaxis, ...])
print(result.shape)

predicted_label_index = np.argmax(result)
print(predicted_label_index)

# to see what is the predicted label
print(image_labels[predicted_label_index])

# load flower dataset
dataset_url = "http://download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url,  cache_dir='.', untar=True)
# cache_dir(where to download data) -->  '.' means current directory
print(data_dir)


data_dir = pathlib.Path(data_dir)
print(data_dir)
# to check flowers are saved correctly
print(list(data_dir.glob('*/*.jpg'))[:5])

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

roses = list(data_dir.glob('roses/*'))
print(roses[:5])

# Reading flowers images from disk into numpy array using opencv
flowers_images_dict = {
    'roses': list(data_dir.glob('roses/*')),
    'daisy': list(data_dir.glob('daisy/*')),
    'dandelion': list(data_dir.glob('dandelion/*')),
    'sunflowers': list(data_dir.glob('sunflowers/*')),
    'tulips': list(data_dir.glob('tulips/*')),
}

# assigning labels to each category
flowers_labels_dict = {
    'roses': 0,
    'daisy': 1,
    'dandelion': 2,
    'sunflowers': 3,
    'tulips': 4,
}


# create test and tran data using the flowers images
X, y = [], []
for flower_name, images in flowers_images_dict.items():
    for image in images:
        img = cv2.imread(str(image))
        resized_img = cv2.resize(img,(224,224))
        X.append(resized_img)
        y.append(flowers_labels_dict[flower_name])

X = np.array(X)
y = np.array(y)

#Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Preprocessing: scale images
X_train_scaled = X_train / 255
X_test_scaled = X_test / 255

# # prediction using pre-trained model on new flowers dataset  --> it predicts wrongly
# print(X[0].shape)
# print(IMAGE_SHAPE+(3,))
# x0_resized = cv2.resize(X[0], IMAGE_SHAPE)
# x1_resized = cv2.resize(X[1], IMAGE_SHAPE)
# x2_resized = cv2.resize(X[2], IMAGE_SHAPE)
# predicted = classifier.predict(np.array([x0_resized, x1_resized, x2_resized]))
# predicted = np.argmax(predicted, axis=1)
# print(predicted)
# print(image_labels[predicted[0])  --> if you open the first image, probably the prediction is wrong


# take pre-trained model and retrain it using flowers images
feature_extractor_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
pretrained_model_without_top_layer = hub.KerasLayer(
    feature_extractor_model, input_shape=(224, 224, 3), trainable=False)
num_of_flowers = 5
model = tf.keras.Sequential([
  pretrained_model_without_top_layer,
  tf.keras.layers.Dense(num_of_flowers)
])
model.summary()

model.compile(
  optimizer="adam",
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['acc'])

model.fit(X_train_scaled, y_train, epochs=5)

model.evaluate(X_test_scaled,y_test)
