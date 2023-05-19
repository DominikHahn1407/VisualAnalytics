import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras import Model
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
from keras.preprocessing import image

IMG_DIR = "C:/Users/Dominik Hahn/Documents/GitHub/VisualAnalytics/data/images/VICE/xoKwbbnlxi0.jpg"
model = load_model("xai_test.h5")


def preprocess_img(img_dir):
    img = image.load_img(img_dir, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return preprocess_input(img)


def grad_cam(img_dir, model, layer_name):
    grad_model = Model(inputs=[model.inputs], outputs=[model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        img = preprocess_img(img_dir)
        layer_output, predictions = grad_model(img)
        loss = predictions[:, 0]

    gradients = tape.gradient(loss, layer_output)[0]
    casted_layer_output = tf.cast(layer_output > 0, "float32")
    casted_gradients = tf.cast(gradients > 0, "float32")
    guided_gradients = casted_layer_output * casted_gradients * gradients

    # Remove unnecessary dims
    layer_output = layer_output[0]

    weights = tf.reduce_mean(guided_gradients, axis=(0,1))
    grad_cam = tf.reduce_sum(tf.multiply(weights, layer_output), axis=-1)

    width, height = img.shape[2], img.shape[1]
    heatmap = cv2.resize(grad_cam.numpy(), (width, height))
    counter = heatmap - np.min(heatmap)
    denominator = (heatmap.max() - heatmap.min())

    scaled_heatmap = counter / denominator
    
    return scaled_heatmap

plt.imshow(grad_cam(IMG_DIR, model, "block5_conv3"))