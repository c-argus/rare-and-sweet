import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from tensorflow.keras.models import load_model
from PIL import Image
from src.data_management import load_pkl_file

def plot_predictions_probabilities(pred_proba, pred_class):
    """
    Visualize the prediction probabilities for each class.
    """
    prob_per_class = pd.DataFrame(
        data=[pred_proba, 1 - pred_proba],
        index=['Healthy', 'Powdery Mildew'],
        columns=['Probability']
    )
    
    prob_per_class['Diagnostic'] = prob_per_class.index

    fig = px.bar(
        prob_per_class,
        x='Diagnostic',
        y='Probability',
        range_y=[0, 1],
        width=600, height=300, template='seaborn'
    )
    st.plotly_chart(fig)

def resize_input_image(img, version):
    """
    Adjust the input image to match the expected model dimensions.
    """
    image_shape = load_pkl_file(file_path=f"outputs/{version}/image_shape.pkl")
    img_resized = img.resize((image_shape[1], image_shape[0]), Image.LANCZOS)
    my_image = np.expand_dims(np.array(img_resized), axis=0) / 255.0

    return my_image

def load_model_and_predict(my_image, version):
    """
    Load the model and make predictions on the input image.
    """
    model = load_model(f"outputs/{version}/cherry_leaves_model.h5")

    # Ensure the image array is correctly formatted
    if my_image is None or not isinstance(my_image, np.ndarray):
        raise ValueError("The input image is not correctly formatted.")

    pred_proba = model.predict(my_image)[0, 0]
    
    target_map = {0: 'Healthy', 1: 'Powdery Mildew'}
    pred_class = target_map[pred_proba > 0.5]

    if pred_class == 'Healthy':
        pred_proba = 1 - pred_proba

    st.write(
        f"The predictive analysis indicates that the cherry leaf is "
        f"**{pred_class.lower()}**."
    )

    return pred_proba, pred_class

