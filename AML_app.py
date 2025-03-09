import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import joblib
from huggingface_hub import hf_hub_download, login
import dill

# (Optional) Enable recursive serialization with dill
dill.settings['recurse'] = True

# --- Custom Function Definitions Start ---
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier
from sklearn.preprocessing import LabelEncoder

# You need to set these values exactly as during training
input_dim_value = 8  # Update to your actual input dimension
le = LabelEncoder()
le.classes_ = ["RUNX1-RUNX1T1", "CBFB-MYH11", "NPM1", "PML-RARA", "Control Group"]  # Update to your actual classes

# Login to Hugging Face and load models
model_path = hf_hub_download(repo_id="rhea-chainani/aml_single_blood_cell", filename="model_v2.keras")
single_cell_model = tf.keras.models.load_model(model_path, compile=False)
dill.settings['recurse'] = True
with open("voting_model.pkl", "rb") as file:
    voting_model = dill.load(file)

# Blood cell type labels
cell_types = [
    "Basophil", "Eosinophil", "Erythroblast", "IG", "Lymphocyte", "Monocyte", "Neutrophil", "Platelet"
]

# AML subtype labels
aml_subtypes = ["RUNX1-RUNX1T1", "CBFB-MYH11", "NPM1", "PML-RARA", "Control Group"]

def preprocess_image(image):
    """
    Converts an image (read with OpenCV) to RGB, resizes it using bicubic interpolation,
    and adds a batch dimension, without normalizing the pixel values.
    """
    # Convert image from BGR (OpenCV default) to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize using bicubic interpolation
    image = cv2.resize(image, (144,144), interpolation=cv2.INTER_CUBIC)
    # Return with an added batch dimension
    return np.expand_dims(image, axis=0)

def classify_cells(images):
    counts = np.zeros(len(cell_types))
    for img in images:
        processed_img = preprocess_image(img)
        prediction = single_cell_model.predict(processed_img)
        cell_type_idx = np.argmax(prediction)
        counts[cell_type_idx] += 1
    return counts

def predict_aml_subtype(counts):
    normalized_counts = counts / np.sum(counts)  # Normalize counts for voting model
    subtype_prediction = voting_model.predict(np.expand_dims(normalized_counts, axis=0))
    subtype_idx = np.argmax(subtype_prediction)
    return aml_subtypes[subtype_idx]

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["About AML", "Classify Blood Cells"])
    
    if page == "About AML":
        st.title("Acute Myeloid Leukemia (AML)")
        st.write("""
        Acute Myeloid Leukemia (AML) is a fast-progressing cancer of the blood and bone marrow, primarily affecting white blood cells. Our application classifies AML into four genetically defined subtypes and a control group, each with unique clinical characteristics that guide treatment decisions:

**RUNX1-RUNX1T1:** This subtype involves a chromosomal translocation that fuses the RUNX1 and RUNX1T1 genes. It typically presents in younger AML patients and is associated with favorable treatment responses.
**CBFB-MYH11:** Characterized by the fusion of the CBFB and MYH11 genes, this subtype is also often seen in younger patients and has a relatively positive prognosis with targeted therapies.
**NPM1:** A common mutation in AML, the NPM1 subtype is characterized by mutations in the NPM1 gene. While generally responsive to treatment, its prognosis can vary based on additional genetic factors.
**PML-RARA:** This subtype results from the fusion of the PML and RARA genes, leading to a distinct form of AML known as Acute Promyelocytic Leukemia (APL). It has a highly specific treatment protocol and, if treated promptly, can have an excellent prognosis.
**Control Group:** This group includes samples without AML, serving as a baseline for comparison and ensuring accuracy in the classification of AML subtypes.
        """)
    
    elif page == "Classify Blood Cells":
        st.title("AML Subtype Classification")
        uploaded_files = st.file_uploader("Upload Blood Cell Images", type=["jpg", "jpeg", "png", "tiff"], accept_multiple_files=True)
        
        if uploaded_files:
            images = [cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1) for file in uploaded_files]
            
            # Count blood cell types
            cell_counts = classify_cells(images)
            
            # Display counts
            st.write("### Count of Blood Cell Types")
            count_dict = {cell_types[i]: int(cell_counts[i]) for i in range(len(cell_types))}
            st.json(count_dict)
            
            # Predict AML subtype
            predicted_subtype = predict_aml_subtype(cell_counts)
            st.write("### Predicted AML Subtype:", predicted_subtype)
            
if __name__ == "__main__":
    main()
