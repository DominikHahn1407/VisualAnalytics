import base64
import os
import streamlit as st

from PIL import Image

st.set_page_config(page_title="Visual Analytics", page_icon=":guardsman:", layout="wide")

with open('style.css') as f:
 st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)


BASE_PATH = os.path.join(os.getcwd(), "results")
model_list = os.listdir(BASE_PATH)
default_model = "InceptionResnet"

IMG_PATH = os.path.join(os.getcwd(), "data/images/3Blue1Brown/3d6DsjIBzJ4.jpg")
img = Image.open(IMG_PATH)

tab1, tab2, tab3 = st.tabs(["Data", "Model", "Prediction"])

with tab2:
    selected_model = st.selectbox("Choose your Model", model_list, index=model_list.index(default_model))
    SELECTED_PATH = os.path.join(BASE_PATH, selected_model)

    confusion_matrix = Image.open(os.path.join(SELECTED_PATH, "confusion_matrix.png"))
    structure = Image.open(os.path.join(SELECTED_PATH, "structure.png"))
    train_acc = Image.open(os.path.join(SELECTED_PATH, "training_acc.png"))
    train_loss = Image.open(os.path.join(SELECTED_PATH, "training_loss.png"))

    st.markdown(f"<h1>Analysis of: {selected_model}</h1>", unsafe_allow_html=True)

    tab2_col1_h1, tab2_col2_h1 = st.columns(2)

    with tab2_col1_h1:
        st.write("Confusion Matrix")
        st.image(confusion_matrix, caption="Confusion Matrix")
    with tab2_col2_h1:
        st.write("Model Structure")
        st.image(structure, caption="Model Structure")

    tab2_col1_h2, tab2_col2_h2 = st.columns(2)

    with tab2_col1_h2:
        st.write("Training Accuracy")
        st.image(train_acc, caption="Training Accuracy")
    with tab2_col2_h2:
        st.write("Training Loss")
        st.image(train_loss, caption="Training Loss")