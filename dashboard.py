import os
import pickle
import streamlit as st
import numpy as np

from PIL import Image


BASE_PATH = os.path.join(os.getcwd(), "results")
model_list = os.listdir(BASE_PATH)
default_model = "InceptionResnet"

IMG_PATH = os.path.join(os.getcwd(), "data/images/3Blue1Brown/3d6DsjIBzJ4.jpg")
img = Image.open(IMG_PATH)

selected_model = st.selectbox("Choose your Model", model_list, index=model_list.index(default_model))
SELECTED_PATH = os.path.join(BASE_PATH, selected_model)

confusion_matrix = Image.open(os.path.join(SELECTED_PATH, "confusion_matrix.png"))
structure = Image.open(os.path.join(SELECTED_PATH, "structure.png"))
train_acc = Image.open(os.path.join(SELECTED_PATH, "training_acc.png"))
train_loss = Image.open(os.path.join(SELECTED_PATH, "training_loss.png"))

st.image(confusion_matrix, caption="Confusion Matrix")
st.image(structure, caption="Model Structure")
st.image(train_acc, caption="Training Accuracy")
st.image(train_loss, caption="Training Loss")