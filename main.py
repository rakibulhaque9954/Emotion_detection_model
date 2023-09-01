import tensorflow as tf
import cv2
from keras.models import load_model
import numpy as np

CLASS_NAMES = ['angry', 'happy', 'sad']
model = load_model('trained_model.h5 copy')
# print(model.summary())
test_image = cv2.imread(0)

# converting image to tensor
image = tf.expand_dims(tf.constant(test_image, dtype = tf.float32), axis= 0)

# expanding dimensions or batching # axis = 0 means adding dimesion along the front for instance: (224, 224, 3) -> (0, 224, 224, 3) batched tensor


print(model(image))