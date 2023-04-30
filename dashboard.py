import os
# import streamlit as st
import numpy as np

from PIL import Image
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input


TEST_DIR = os.path.join(os.getcwd(), "data/test")
TARGET_SIZE = (224, 224)

model = load_model("./Inception_0933.h5")
dir = os.listdir(TEST_DIR)
rand_int = np.random.randint(len(dir))
image_path = os.path.join(TEST_DIR, f"test_{rand_int}.png")

img = Image.open(image_path)
img = np.array(img, dtype=np.uint8)
img_processed = preprocess_input(img)

print(img_processed.shape)