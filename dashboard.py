import os
import pickle
import streamlit as st
import numpy as np

from PIL import Image
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input


with open('le_classes.pkl', 'rb') as f:
    classes = pickle.load(f)

IMG_PATH = os.path.join(os.getcwd(), "data/images/3Blue1Brown/3d6DsjIBzJ4.jpg")
TARGET_SIZE = (224, 224)

model = load_model("./Inception_0933.h5")

img = Image.open(IMG_PATH)
rgb_image = img.copy().convert("RGB")
img_prepared = np.array(rgb_image.resize(TARGET_SIZE), dtype=np.uint8)
img_prepared = img_prepared[np.newaxis, :, :]

pred = model.predict(img_prepared)
prediction = classes[np.argmax(pred, axis=1)]

st.image(img, caption="Image")
st.write(prediction)