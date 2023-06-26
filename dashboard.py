import csv
import os
import pickle
import matplotlib.pyplot as plt
import streamlit as st

from PIL import Image


st.set_option('deprecation.showPyplotGlobalUse', False)
# Set the page title and the layout
st.set_page_config(page_title="Visual Analytics", page_icon=":guardsman:", layout="wide")


# with open('style.css') as f:
#     st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)

# Write a list for all used models in this project
MODEL_LIST = ["Custom", "VGG_16", "Inception", "ResNet", "Xception"]

# Create the path structure, where the images are stored
BASE_PATH = os.path.join(os.getcwd(), "results")
CLASS_ACC_PATH = os.path.join(BASE_PATH, "Class_Accuracy")
DATA_PATH = os.path.join(BASE_PATH, "Data")
MODEL_PATH = os.path.join(BASE_PATH, "Model")
XAI_PATH = os.path.join(BASE_PATH, "XAI")
IMAGE_PATH = os.path.join(DATA_PATH, "Images")
CSV_PATH = os.path.join(DATA_PATH, "CSV")

# Write an title for the dashboard
st.markdown(f"<h1 style='text-align: center;'>Classifying YouTube Thumbnails</h1>", unsafe_allow_html=True)

# Create a dropdown for the possibility to select the version of the dataset (v1 with all classes) or (v2 with 7 clustered classes)
version_list = ["v1", "v2"]
version_selection = st.selectbox("Choose the Dataset", version_list)
MODEL_PATH = os.path.join(MODEL_PATH, version_selection)

image_list = os.listdir(IMAGE_PATH)

# Read in the distribution of the specific version of the dataset and create a dict with categories and the specific occurences to create a bar chart
occurence_dict = {}
with open(os.path.join(CSV_PATH, f"distribution_{version_selection}.csv"), "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        category = row['Category']
        occurence = int(row['Count'])
        occurence_dict[category] = occurence

# Read in the exported accuracies for each model (Mean-Accuracy and Weighted Accuracy)
with open("accuracies_v1.pkl", "rb") as f:
    accuracies_v1 = pickle.load(f)
with open("accuracies_v2.pkl", "rb") as f:
    accuracies_v2 = pickle.load(f)

# Read in the predictions of all models for the 5 diesplayed images
with open("predictions.pkl", "rb") as f:
    all_predictions = pickle.load(f)

accuracy_dict = [accuracies_v1, accuracies_v2]


# Create a tab for the data analysis, the model analysis and the combination of both
tab1, tab2, tab3 = st.tabs(["Data", "Model", "Prediction"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        # Create a dropdown for the possible images
        selected_image = st.selectbox("Choose an specific Image", image_list)
        selected_value = selected_image.split(" ")[version_list.index(version_selection)]

    with col2:
        # Create a dropdown for the possibility to show the standard augmentation (used for the training) or the exotic augmentation (all possibilities)
        exotic = st.selectbox("Choose an Augmentation Version", ["Standard", "Exotic"])

    # Load an resize the origial image and the augmented image
    IMG_PATH = os.path.join(IMAGE_PATH, selected_image)
    image = Image.open(os.path.join(IMG_PATH, "original.jpg"))
    image = image.resize((640,500))
    augmented_img = Image.open(os.path.join(IMG_PATH, f"{exotic.lower()}.png"))
    augmented_img = augmented_img.resize((640,500))
 
    # From the dictionary with the distributions and categories create a barchart in matplotlib to show the data distribution for each class
    distribution_list = [(key, value) for key, value in occurence_dict.items()]
    distribution_list.sort(key=lambda x: x[1], reverse=True)
    categories = [x[0] for x in distribution_list]
    occurences = [x[1] for x in distribution_list]

    fig, ax = plt.subplots()

    ax.bar(categories, occurences, color='#ff4122')
    ax.set_xticklabels(categories, rotation=90)
    ax.set_xlabel("Categories")
    ax.set_ylabel("Occurences")
    plt.tight_layout()

    DIAGRAMM_PATH= os.path.join(MODEL_PATH, "occurence.png")
    plt.savefig(DIAGRAMM_PATH)
    plt.close()
    occurence = Image.open(DIAGRAMM_PATH)
    occurence = occurence.resize((640,500))

    col1, col2, col3 = st.columns(3)
    with col1:
        # Display the original image
        st.markdown(f"<h5 style='text-align: center;'>Image for Category:  {selected_value}</h5>", unsafe_allow_html=True)
        st.image(image)

    with col2:
        # Display the data distribution
        st.markdown("<h5 style='text-align: center;'>Data Distribution </h5>", unsafe_allow_html=True)
        st.image(occurence)
        
    with col3:
        # Display the augmented version of the image
        st.markdown("<h5 style='text-align: center;'>Augmented Version</h5>", unsafe_allow_html=True)
        st.image(augmented_img, use_column_width="always")
        
        

with tab2:
    # Create a dropdown with the possible images
    selected_model = st.selectbox("Choose your Model", MODEL_LIST)
    # Load in and resize the confusion matrix of the specific model
    confusion_matrix = Image.open(os.path.join(MODEL_PATH, f"{selected_model.lower()}_cm.png"))
    # Load in the train and val accuracy of the trainings history for the specific model
    train_acc = Image.open(os.path.join(MODEL_PATH, f"{selected_model.lower()}_acc.png"))
    # load in the train and val loss of the training history for the specific model
    train_loss = Image.open(os.path.join(MODEL_PATH, f"{selected_model.lower()}_loss.png"))
    # Load in the structure of the specific model
    model_vis = Image.open(os.path.join(MODEL_PATH,  f"{selected_model.lower()}_model.png"))
    

    # Ass the structure of the models for custom and vgg16 is very small, the images should be displayed in one row
    if selected_model == "Custom" or selected_model == "VGG_16":
        tab2_col1, tab2_col2, tab2_col3, tab2_col4 = st.columns(4)
    
        with tab2_col1:
            st.markdown("<h5 style='text-align: center;'> Confusion Matrix </h5>", unsafe_allow_html=True)
            st.image(confusion_matrix.resize((651, 580)))     
        with tab2_col2:
            st.markdown("<h5 style='text-align: center;'>Training Loss </h5>", unsafe_allow_html=True)
            st.image(train_loss.resize((651, 580)))
        with tab2_col3:
            st.markdown("<h5 style='text-align: center;'>Training Accuracy </h5>", unsafe_allow_html=True)
            st.image(train_acc.resize((651, 580)),use_column_width="auto")
        with tab2_col4:
            st.markdown("<h5 style='text-align: center;'>Vizualized Modelstructure </h5>", unsafe_allow_html=True)
            st.image(model_vis.resize((651, 580)), use_column_width="auto")
    # For the other models, the structure should be displayed beneath the other images
    else:
        tab2_col1, tab2_col2, tab2_col3 = st.columns(3)
        with tab2_col1:
            st.markdown("<h5 style='text-align: center;'> Confusion Matrix </h5>", unsafe_allow_html=True)
            st.image(confusion_matrix.resize((1900, 1000)), use_column_width="auto")
        with tab2_col2:
            st.markdown("<h5 style='text-align: center;'>Training Loss </h5>", unsafe_allow_html=True)
            st.image(train_loss.resize((1900, 1000)), use_column_width="auto")
        with tab2_col3:
            st.markdown("<h5 style='text-align: center;'>Training Accuracy </h5>", unsafe_allow_html=True)
            st.image(train_acc.resize((1900, 1000)), use_column_width="auto")

        st.markdown("<h5 style='text-align: center;'>Vizualized Modelstructure </h5>", unsafe_allow_html=True)
        st.image(model_vis, use_column_width=True)


with tab3:
    col1_h1, col2_h1 = st.columns(2)
    with col1_h1:
        # As folder structure is not the same as image training/saving structure, we need this little workaround
        value_list = [1, 4, 3, 0, 2]
        image_list_prepared = []
        for index, item in enumerate(value_list):
            image_list_prepared.append(f"{item} {image_list[index]}")

        image_list_prepared.sort()

        # Create a dropdown to select the possible image which should be analyzed
        selected_image_pred = st.selectbox("Choose an specific Image", image_list_prepared, key="selected_image_pred")
        selected_image_index = selected_image_pred[0]
        selected_image_pred = selected_image_pred[2:]
        selected_value = selected_image_pred.split(" ")[version_list.index(version_selection)]

    with col2_h1:        
        # Create a dropdown to choose a model, which should be used to analyse the chosen image
        selected_model_pred = st.selectbox("Choose your Model", MODEL_LIST, key="selected_model_pred")

    col1_h2, col2_h2, col3_h2 = st.columns(3)
    with col1_h2:
        # Load in the originial raw image input
        st.markdown("<h5 style='text-align: center;'>Original Image</h5>", unsafe_allow_html=True)
        IMG_PATH = os.path.join(IMAGE_PATH, selected_image_pred)
        image = Image.open(os.path.join(IMG_PATH, "original.jpg"))
        st.image(image.resize((950, 500)), use_column_width="auto")

        # Load in the grad cam image for the model for the specific input image
        grad_value = f"{selected_model_pred.lower()}_{selected_image_index}.png"
        grad_path = os.path.join(XAI_PATH, "grad", version_selection, grad_value)
        grad_image = Image.open(grad_path)

        st.markdown("<h5 style='text-align: center;'>Grad CAM</h5>", unsafe_allow_html=True)
        st.image(grad_image.resize((950, 500)), use_column_width="auto")

    with col2_h2:
        model_value = None
        if selected_model_pred == "Inception":
            model_value = "inception_v3"
        elif selected_model_pred == "ResNet":
            model_value = "resnet_50"
        else:
            model_value = selected_model_pred.lower()

        # Read in the accuracies for the specific models (mean and weighted acc)
        accuracy_dict = accuracy_dict[version_list.index(version_selection)]
        accuracies = accuracy_dict[model_value]        
        
        # Read in the predictions for the specific models
        prediction_dict = all_predictions[version_list.index(version_selection)]
        predictions = prediction_dict[selected_model_pred.lower()]
        prediction = predictions[int(selected_image_index)]
        prediction = prediction[0].upper() + prediction[1:]
        
        st.markdown("<h5 style='text-align: center; margin-bottom: 10%'> </h5>", unsafe_allow_html=True)
        # Display the prediction of the model for the specific image
        st.markdown(f"<h4 style='text-align: center;'>Prediction: {prediction}", unsafe_allow_html=True)
        # Display the True Label of the chosen image
        st.markdown(f"<h4 style='text-align: center;'>True Label: {selected_value}", unsafe_allow_html=True)
        # Display the mean accuracy of the model on the test set
        st.markdown(f"<h4 style='text-align: center;'>Accuracy: {accuracies[0]}</h4>", unsafe_allow_html=True)
        # Display the weighted accuracy of the model on the test set
        st.markdown(f"<h4 style='text-align: center; margin-bottom: 10%'>Weighted Accuracy: {accuracies[1]}</h4>", unsafe_allow_html=True)

        # load in lime images for the chosen model on the specific image
        lime_value = f"{selected_model_pred.lower()}_{selected_image_index}.png"
        lime_path = os.path.join(XAI_PATH, "lime", version_selection, lime_value)
        lime_image = Image.open(lime_path)

        st.markdown("<h5 style='text-align: center;'>Lime</h5>", unsafe_allow_html=True)
        st.image(lime_image.resize((950, 500)), use_column_width="auto")

    with col3_h2:
        # load in the class accuracies for the specific model on the test set
        class_acc_value = f"{selected_model_pred.lower()}.png"
        class_acc_path = os.path.join(CLASS_ACC_PATH, version_selection, class_acc_value)
        class_acc_image = Image.open(class_acc_path)

        # load in the shap values for the specific model on the chosen image
        shap_value = f"{selected_model_pred.lower()}_{selected_image_index}.png"
        shap_path = os.path.join(XAI_PATH, "shap", version_selection, shap_value)
        shap_image = Image.open(shap_path)

        st.markdown("<h5 style='text-align: center;'>Class Accuracies</h5>", unsafe_allow_html=True)
        st.image(class_acc_image.resize((950, 500)), use_column_width="auto")

        st.markdown("<h5 style='text-align: center;'>Shapley Values</h5>", unsafe_allow_html=True)
        st.image(shap_image.resize((950, 500)), use_column_width="auto")

        

   

    