import csv
import os
import matplotlib.pyplot as plt
import streamlit as st

from PIL import Image

st.set_page_config(page_title="Visual Analytics", page_icon=":guardsman:", layout="wide")

with open('style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)

BASE_PATH = os.path.join(os.getcwd(), "results")
MODEL_PATH = os.path.join(BASE_PATH, "Model")
DATA_PATH = os.path.join(BASE_PATH, "Data")
IMAGE_PATH = os.path.join(DATA_PATH, "Images")
CSV_PATH = os.path.join(DATA_PATH, "CSV")

version_selection = st.selectbox("Choose the Dataset", ["v1", "v2"])
MODEL_PATH = os.path.join(MODEL_PATH, version_selection)

model_list = os.listdir(MODEL_PATH)
image_list = os.listdir(IMAGE_PATH)

occurence_dict = {}
with open(os.path.join(CSV_PATH, f"distribution_{version_selection}.csv"), "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        category = row['Category']
        occurence = int(row['Count'])
        occurence_dict[category] = occurence
# CSS-Styling für die Navbar

tab1, tab2, tab3 = st.tabs(["Data", "Model", "Prediction"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        selected_image = st.selectbox("Choose an specific Image", image_list)
    
    IMG_PATH = os.path.join(IMAGE_PATH, selected_image)
    image = Image.open(os.path.join(IMG_PATH, os.listdir(IMG_PATH)[0]))
    # st.markdown(f"<h1>Image for Category: {selected_image}</h1>", unsafe_allow_html=True)
    # st.image(image)

    distribution_list = [(key, value) for key, value in occurence_dict.items()]
    distribution_list.sort(key=lambda x: x[1], reverse=True)
    categories = [x[0] for x in distribution_list]
    occurences = [x[1] for x in distribution_list]

    fig, ax = plt.subplots()
    ax.bar(categories, occurences)
    ax.set_xticklabels(categories, rotation=90)
    ax.set_xlabel("Categories")
    ax.set_ylabel("Occurences")
    # ax.set_title("Data Distribution")
    # st.pyplot(fig)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<h1 style='text-align: center;'>Image for Category: <br> {selected_image}</h1>", unsafe_allow_html=True)
        st.image(image)
        st.markdown("<h1 style='text-align: center;'>Augmented Version</h1>", unsafe_allow_html=True)


    with col2:
        st.markdown("<h1 style='text-align: center;'>Data <br> Distribution </h1>", unsafe_allow_html=True)
        st.pyplot(fig) 


with tab2:
    selected_model = st.selectbox("Choose your Model", model_list)
    SELECTED_PATH = os.path.join(MODEL_PATH, selected_model)

    confusion_matrix = Image.open(os.path.join(SELECTED_PATH, "confusion_matrix.png"))
    # structure = Image.open(os.path.join(SELECTED_PATH, "structure.png"))
    train_acc = Image.open(os.path.join(SELECTED_PATH, "hist_acc.png"))
    train_loss = Image.open(os.path.join(SELECTED_PATH, "hist_loss.png"))

    st.markdown(f"<h1>Analysis of: {selected_model}</h1>", unsafe_allow_html=True)

    tab2_col1_h1, tab2_col2_h1 = st.columns(2)

    with tab2_col1_h1:
        st.write("Confusion Matrix")
        st.image(confusion_matrix, caption="Confusion Matrix")
        st.markdown("<h1 style='text-align: center;'>Visualisierung des Models</h1>", unsafe_allow_html=True)
    # with tab2_col2_h1:
    #     st.write("Model Structure")
    #     st.image(structure, caption="Model Structure")

    tab2_col1_h2, tab2_col2_h2 = st.columns(2)

    with tab2_col1_h2:
        st.write("Training Accuracy")
        st.image(train_acc, caption="Training Accuracy")
    with tab2_col2_h2:
        st.write("Training Loss")
        st.image(train_loss, caption="Training Loss")

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h1 style='text-align: center;'>Original Bild</h1>", unsafe_allow_html=True)
        

    with col2:
        st.markdown("<h1 style='text-align: center;'>Barchart Prediction (Confidences)</h1>", unsafe_allow_html=True)


    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<h1 style='text-align: center;'>Grad CAM</h1>", unsafe_allow_html=True)
    with col2:
        st.markdown("<h1 style='text-align: center;'>Lime</h1>", unsafe_allow_html=True)
    with col3:
        st.markdown("<h1 style='text-align: center;'>Shep</h1>", unsafe_allow_html=True)

    