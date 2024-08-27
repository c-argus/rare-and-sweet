from PIL import Image
import numpy as np
import pandas as pd
import streamlit as st

# Import functions from your custom modules for handling downloads, predictions, and plotting
from src.data_management import download_dataframe_as_csv
from src.machine_learning.predictive_analysis import (
    load_model_and_predict,
    resize_input_image,
    plot_predictions_probabilities
)

def resize_input_image(img, version):
    """
    Adjust the input image to match the expected model dimensions of 160x160 pixels.
    """
    image_shape = (112, 112)  # Update to 112x112 dimensions
    img_resized = img.resize((image_shape[1], image_shape[0]), Image.LANCZOS)
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize and add batch dimension
    return img_array

# Define the main function for the cherry leaves detector page
def page_cherry_leaves_detector_body():
    st.title('Powdery Mildew Detection in Cherry Leaves')
    st.info(
        f"* The client is interested in determining whether a given cherry leaf is healthy "
        f"or affected by powdery mildew."
    )

    st.write(
        f"* You can upload images of cherry leaves for live prediction. "
        f"You can download a sample dataset from [here](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves)."
    )

    st.write("---")

    images_buffer = st.file_uploader('Upload cherry leaf images. You may select more than one.',
                                     accept_multiple_files=True)

    if images_buffer is not None:
        df_report = pd.DataFrame([])
        for image in images_buffer:
            img_pil = Image.open(image)
            st.info(f"Cherry Leaf Sample: **{image.name}**")
            img_array = np.array(img_pil)
            st.image(img_pil, caption=f"Image Size: {img_array.shape[1]}px width x {img_array.shape[0]}px height")

            version = 'v1'
            resized_img = resize_input_image(img=img_pil, version=version)
            pred_proba, pred_class = load_model_and_predict(resized_img, version=version)
            plot_predictions_probabilities(pred_proba, pred_class)

            df_report = df_report.append({"Name": image.name, 'Result': pred_class}, ignore_index=True)
        
        if not df_report.empty:
            st.success("Analysis Report")
            st.table(df_report)
            st.markdown(download_dataframe_as_csv(df_report), unsafe_allow_html=True)



