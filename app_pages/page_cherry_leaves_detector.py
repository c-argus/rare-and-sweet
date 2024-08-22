import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd

# Import functions from your custom modules for handling downloads, predictions, and plotting
from src.data_management import download_dataframe_as_csv
from src.machine_learning.predictive_analysis import (
                                                    load_model_and_predict,
                                                    resize_input_image,
                                                    plot_predictions_probabilities
                                                    )

# Define the main function for the cherry leaves detector page
def page_cherry_leaves_detector_body():
    st.title('Powdery Mildew Detection in Cherry Leaves')
    # Display informational text to the user about the purpose of this page
    st.info(
        f"* The client is interested in determining whether a given cherry leaf is healthy "
        f"or affected by powdery mildew."
    )

    # Instructions for the user to upload images and a link to download a sample dataset
    st.write(
        f"* You can upload images of cherry leaves for live prediction. "
        f"You can download a sample dataset from [here](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves)."
    )

    st.write("---")

    # File uploader widget where users can upload multiple images
    images_buffer = st.file_uploader('Upload cherry leaf images. You may select more than one.',
                                     type='png', accept_multiple_files=True)
   
    # Check if any images have been uploaded
    if images_buffer is not None:
        df_report = pd.DataFrame([])
        for image in images_buffer:

            img_pil = Image.open(image)
            st.info(f"Cherry Leaf Sample: **{image.name}**")
            img_array = np.array(img_pil)
            st.image(img_pil, caption=f"Image Size: {img_array.shape[1]}px width x {img_array.shape[0]}px height")

            version = 'v1'
            # Resize the uploaded image according to the model's requirements
            resized_img = resize_input_image(img=img_pil, version=version)
            pred_proba, pred_class = loaxdd_model_and_predict(resized_img, version=version)
            plot_predictions_probabilities(pred_proba, pred_class)

            # Append the results (image name and predicted class) to the DataFrame
            df_report = df_report.append({"Name": image.name, 'Result': pred_class},
                                         ignore_index=True)
        
        # If there are any results, display them as a table and provide a download link
        if not df_report.empty:
            st.success("Analysis Report")
            st.table(df_report)
            st.markdown(download_dataframe_as_csv(df_report), unsafe_allow_html=True)
