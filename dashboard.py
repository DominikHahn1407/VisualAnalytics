import csv
import os
import pickle
import matplotlib.pyplot as plt
import streamlit as st

from PIL import Image

st.set_page_config(page_title="Visual Analytics", page_icon=":guardsman:", layout="wide")

with open('style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)

MODEL_LIST = ["Custom", "VGG_16", "Inception", "ResNet", "Xception"]

BASE_PATH = os.path.join(os.getcwd(), "results")

CLASS_ACC_PATH = os.path.join(BASE_PATH, "Class_Accuracy")
DATA_PATH = os.path.join(BASE_PATH, "Data")
MODEL_PATH = os.path.join(BASE_PATH, "Model")
XAI_PATH = os.path.join(BASE_PATH, "XAI")

IMAGE_PATH = os.path.join(DATA_PATH, "Images")
CSV_PATH = os.path.join(DATA_PATH, "CSV")

version_list = ["v1", "v2"]
version_selection = st.selectbox("Choose the Dataset", version_list)
MODEL_PATH = os.path.join(MODEL_PATH, version_selection)

image_list = os.listdir(IMAGE_PATH)

occurence_dict = {}
with open(os.path.join(CSV_PATH, f"distribution_{version_selection}.csv"), "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        category = row['Category']
        occurence = int(row['Count'])
        occurence_dict[category] = occurence

with open("accuracies_v1.pkl", "rb") as f:
    accuracies_v1 = pickle.load(f)

with open("accuracies_v2.pkl", "rb") as f:
    accuracies_v2 = pickle.load(f)

with open("predictions.pkl", "rb") as f:
    all_predictions = pickle.load(f)

accuracy_dict = [accuracies_v1, accuracies_v2]

# CSS-Styling für die Navbar

tab1, tab2, tab3 = st.tabs(["Data", "Model", "Prediction"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        selected_image = st.selectbox("Choose an specific Image", image_list)
        selected_value = selected_image.split(" ")[version_list.index(version_selection)]

    with col2:
        exotic = st.selectbox("Choose an Augmentation Version", ["Standard", "Exotic"])

    IMG_PATH = os.path.join(IMAGE_PATH, selected_image)
    image = Image.open(os.path.join(IMG_PATH, "original.jpg"))
    augmented_img = Image.open(os.path.join(IMG_PATH, f"{exotic.lower()}.png"))
    # st.markdown(f"<h1>Image for Category: {selected_image}</h1>", unsafe_allow_html=True)
    # st.image(image)

    distribution_list = [(key, value) for key, value in occurence_dict.items()]
    distribution_list.sort(key=lambda x: x[1], reverse=True)
    categories = [x[0] for x in distribution_list]
    occurences = [x[1] for x in distribution_list]

    fig, ax = plt.subplots()
    ax.bar(categories, occurences, color='#ff4122')
    ax.set_xticklabels(categories, rotation=90)
    ax.set_xlabel("Categories")
    ax.set_ylabel("Occurences")
    # ax.set_title("Data Distribution")
    # st.pyplot(fig)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<h1 style='text-align: center;'>Image for Category: <br> {selected_value}</h1>", unsafe_allow_html=True)
        st.image(image)
        st.markdown("<h1 style='text-align: center;'>Augmented Version</h1>", unsafe_allow_html=True)
        st.image(augmented_img)

    with col2:
        st.markdown("<h1 style='text-align: center;'>Data <br> Distribution </h1>", unsafe_allow_html=True)
        st.pyplot(fig) 


with tab2:
    selected_model = st.selectbox("Choose your Model", MODEL_LIST)
    tab2_col1, tab2_col2, tab2_col3, tab2_col4 = st.columns(4)
    
    confusion_matrix = Image.open(os.path.join(MODEL_PATH, f"{selected_model.lower()}_cm.png"))
    train_acc = Image.open(os.path.join(MODEL_PATH, f"{selected_model.lower()}_acc.png"))
    train_loss = Image.open(os.path.join(MODEL_PATH, f"{selected_model.lower()}_loss.png"))
    model_vis = Image.open(os.path.join(MODEL_PATH,  f"{selected_model.lower()}_model.png"))
    with tab2_col1:
        # st.markdown("<h1 style='text-align: center;'>Visualisierung des Models</h1>", unsafe_allow_html=True)
        st.image(confusion_matrix, caption="Confusion Matrix")
    with tab2_col2:
        st.image(train_loss, caption="Training Loss")
    with tab2_col3:
        st.image(train_acc, caption="Training Accuracy")
    with tab2_col4:
        st.image(model_vis, caption="Vizualized Modelstructure")


    # confusion_matrix = Image.open(os.path.join(MODEL_PATH, f"{selected_model.lower()}_cm.png"))
    # structure = Image.open(os.path.join(SELECTED_PATH, "structure.png"))
    # train_acc = Image.open(os.path.join(MODEL_PATH, f"{selected_model.lower()}_acc.png"))
    # train_loss = Image.open(os.path.join(MODEL_PATH, f"{selected_model.lower()}_loss.png"))

    # st.markdown(f"<h1>Analysis of: {selected_model}-Model</h1>", unsafe_allow_html=True)

    # tab2_col1_h1, tab2_col2_h1 = st.columns(2)

    # with tab2_col1_h1:
        # st.write("Confusion Matrix")
        
    # with tab2_col2_h1:
    #     st.write("Model Structure")
    #     st.image(structure, caption="Model Structure")

    # tab2_col1_h2, tab2_col2_h2 = st.columns(2)

    # with tab2_col1_h2:
    #     # st.write("Training Accuracy")
    #     st.image(train_acc, caption="Training Accuracy")
    # with tab2_col2_h2:
    #     # st.write("Training Loss")
    #     st.image(train_loss, caption="Training Loss")

with tab3:
    col1_h1, col2_h1 = st.columns(2)
    with col1_h1:
        # As folder structure is not the same as image training/saving structure, we need this little workaround
        value_list = [1, 4, 3, 0, 2]
        image_list_prepared = []
        for index, item in enumerate(value_list):
            image_list_prepared.append(f"{item} {image_list[index]}")

        image_list_prepared.sort()

        selected_image_pred = st.selectbox("Choose an specific Image", image_list_prepared, key="selected_image_pred")
        selected_image_index = selected_image_pred[0]
        selected_image_pred = selected_image_pred[2:]
        selected_value = selected_image_pred.split(" ")[version_list.index(version_selection)]

    with col2_h1:        
        selected_model_pred = st.selectbox("Choose your Model", MODEL_LIST, key="selected_model_pred")

    col1_h2, col2_h2, col3_h2 = st.columns(3)
    with col1_h2:
        st.markdown("<h6 style='text-align: center;'>Original Image</h6>", unsafe_allow_html=True)
        IMG_PATH = os.path.join(IMAGE_PATH, selected_image_pred)
        image = Image.open(os.path.join(IMG_PATH, "original.jpg"))
        st.image(image)

        grad_value = f"{selected_model_pred.lower()}_{selected_image_index}.png"
        grad_path = os.path.join(XAI_PATH, "grad", version_selection, grad_value)
        grad_image = Image.open(grad_path)

        st.markdown("<h6 style='text-align: center;'>Grad CAM</h6>", unsafe_allow_html=True)
        st.image(grad_image)

    with col2_h2:
        model_value = None
        if selected_model_pred == "Inception":
            model_value = "inception_v3"
        elif selected_model_pred == "ResNet":
            model_value = "resnet_50"
        else:
            model_value = selected_model_pred.lower()

        accuracy_dict = accuracy_dict[version_list.index(version_selection)]
        accuracies = accuracy_dict[model_value]        
        
        prediction_dict = all_predictions[version_list.index(version_selection)]
        predictions = prediction_dict[selected_model_pred.lower()]
        prediction = predictions[int(selected_image_index)]
        prediction = prediction[0].upper() + prediction[1:]

        st.markdown(f"<h3 style='text-align: center;'>Prediction: {prediction}", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center;'>True Label: {selected_value}", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center;'>Accuracy: {accuracies[0]}</h3>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center;'>Weighted Accuracy: {accuracies[1]}</h3>", unsafe_allow_html=True)

        lime_value = f"{selected_model_pred.lower()}_{selected_image_index}.png"
        lime_path = os.path.join(XAI_PATH, "lime", version_selection, lime_value)
        lime_image = Image.open(lime_path)

        st.markdown("<h6 style='text-align: center;'>Lime</h6>", unsafe_allow_html=True)
        st.image(lime_image)

    with col3_h2:
        class_acc_value = f"{selected_model_pred.lower()}.png"
        class_acc_path = os.path.join(CLASS_ACC_PATH, version_selection, class_acc_value)
        class_acc_image = Image.open(class_acc_path)

        shap_value = f"{selected_model_pred.lower()}_{selected_image_index}.png"
        shap_path = os.path.join(XAI_PATH, "shap", version_selection, shap_value)
        shap_image = Image.open(shap_path)

        st.markdown("<h6 style='text-align: center;'>Class Accuracies</h6>", unsafe_allow_html=True)
        st.image(class_acc_image)

        st.markdown("<h6 style='text-align: center;'>Shapley Values</h6>", unsafe_allow_html=True)
        st.image(shap_image)
   

    