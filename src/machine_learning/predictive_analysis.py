import streamlit as st 
import numpy as np 
import pandas as pd 
import plotly.express as px
from tensorflow.keras.models import load_model
from PIL import Image
from src.data_management import load_pkl_file 

# Function to plot prediction probabilities as a bar graph using Plotly
def plot_predictions_probabilities(pred_proba, pred_class):
    '''
    Plots the probabilistic result of an image being 'infected' or 'healthy' as a bar graph.
    We use Plotly to create an interactive bar chart that shows the probability per class ('Healthy' or 'Infected').
    This function uses Streamlit's plotly_chart to render the plot in the web app.
    
    Parameters:
    pred_proba (float): Probability of the image being 'infected'.
    pred_class (str): Predicted class label ('Healthy' or 'Infected').
    '''

    # Create a DataFrame to store probabilities for each class
    prob_per_class = pd.DataFrame(
        data=[0, 0],  # Initialize both probabilities to 0
        index={'Healthy': 1, 'Infected': 0}.keys(),  # Labels for the rows
        columns=['Probability']  # Column name
    )

    # Update DataFrame with the predicted probability
    prob_per_class.loc[pred_class] = pred_proba

    # Set the probability for the other class to 1 - predicted probability
    for x in prob_per_class.index.to_list():
        if x not in pred_class:
            prob_per_class.loc[x] = 1 - pred_proba

    # Round the probabilities to three decimal places for display
    prob_per_class = prob_per_class.round(3)

    # Add a column to identify the diagnostic category
    prob_per_class["Diagnostic"] = prob_per_class.index

    # Create the bar plot using Plotly
    fig = px.bar(
        prob_per_class,
        x='Diagnostic',  # x-axis: 'Healthy' or 'Infected'
        y='Probability',  # y-axis: predicted probabilities
        range_y=[0, 1],  # Set y-axis range from 0 to 1
        width=600, height=300,  # Set plot dimensions
        template='seaborn'  # Plotly template style
    )

    # Render the plot in the Streamlit app
    st.plotly_chart(fig)

# Function to resize the input image to the required shape for model prediction
def resize_input_image(img, version):
    '''
    Resizes the input image to the average image size expected by the model.
    Loads the average image dimensions from a pickle file to ensure compatibility with the model.
    
    Parameters:
    img (PIL.Image): The input image to be resized.
    version (str): The version of the model, used to locate the image shape file.
    
    Returns:
    np.ndarray: The resized image normalized to a scale of 0 to 1.
    '''
    
    # Load the target image shape from a pickle file
    image_shape = load_pkl_file(file_path=f"outputs/{version}/image_shape.pkl")

    # Resize the image to the required dimensions using LANCZOS resampling
    img_resized = img.resize((image_shape[1], image_shape[0]), Image.LANCZOS)

    # Expand dimensions to match model input shape and normalize pixel values
    my_image = np.expand_dims(img_resized, axis=0) / 255

    return my_image

# Function to load the ML model and predict the class of the input image
def load_model_and_predict(my_image, version):
    '''
    Loads a pre-trained machine learning model and predicts whether the input image is 'healthy' or 'infected'.
    It also calculates the probability associated with the prediction.
    
    Parameters:
    my_image (np.ndarray): The processed image ready for model prediction.
    version (str): The version of the model to be loaded.
    
    Returns:
    tuple: The prediction probability and predicted class label.
    '''
    
    # Load the pre-trained model from the specified path
    model = load_model(f'outputs/{version}/cherry_leaves_model.h5')

    # Predict the probability of the 'infected' class
    pred_proba = model.predict(my_image)[0, 0]

    # Map the numeric prediction to a human-readable label
    target_map = {v: k for k, v in {'Infected': 1, 'Healthy': 0}.items()}
    pred_class = target_map[pred_proba > 0.5]  # Classify based on probability threshold (0.5)

    # Adjust probability if the prediction is 'Healthy'
    if pred_class == target_map[0]:
        pred_proba = 1 - pred_proba

    # Display the prediction result on the Streamlit app
    st.write(
        f'The predictive analysis indicates the leaf image is '
        f'**{pred_class.lower()}**.'
    )

    return pred_proba, pred_class


