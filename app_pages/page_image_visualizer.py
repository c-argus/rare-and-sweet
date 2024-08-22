import streamlit as st
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread

import itertools
import random

# Function that handles the actions triggered by the checkbox selections and 
# displays the relevant visualizations and explanations.

def page_image_visualizer_body():
    st.title("Image Visualizer")
    st.write(
        '* **In response to a business need, we conducted a visual analysis** '
        '**to differentiate between powdery mildew leaves and healthy ones.**'
    )

    version = 'v1'
    dataset_dir = 'inputs/cherry-leaves_dataset/cherry-leaves'  # Ensure this path is correct

    # Option to compare the average and variability between 
    # the infected and healthy leaves
    if st.checkbox('Difference between average and variability image'):

        avg_powdery_mildew_img = plt.imread('outputs/v1/avg_var_powdery_mildew.png')
        avg_healthy_img = plt.imread(f'outputs/{version}/avg_var_healthy.png')

        st.warning(
            'The study identified subtle patterns in the average and variability images, '
            'making it possible to distinguish between infected and healthy leaves. '
            'Powdery mildew samples, in particular, exhibited slight texture differences in the average images.'
        )
        
        st.image(avg_powdery_mildew_img, caption='Powdery Mildew Leaf - Average Variability')
        st.image(avg_healthy_img, caption='Healthy Leaf - Average Variability')
        st.info(
            f'**Mean:**\n\n'
            f'The mean, or average, is determined by adding all the observations together and then dividing by the '
            f'number of observations. It gives a single value that reflects the central point of the data set.\n\n'

            f'**Standard Deviation (SD):**\n\n'
            f'Standard deviation assesses the dispersion of data points around the mean. It is a measure of the '
            f'variability within the sample.\n\n'
            f'Descriptive statistics, such as central tendency and spread, are essential for summarizing the dataset. '
            f'The *mean* shows the central value, while the *standard deviation* indicates how spread out the values are '
            f'around the mean. Presenting both gives a full picture of the data.'
        )
        st.write('---')

    # Option to compare the average between parasitized and uninfected cells
    if st.checkbox('Differences between average parasitised and average uninfected cells'):
        avg_diff_image = plt.imread(f'outputs/{version}/avg_diff.png')

        st.warning(
            '* The study uncovered subtle differences in patterns, helping to intuitively distinguish between samples.'
        )
        st.image(avg_diff_image, caption='Difference Between Average Images')

    # Option to create an image montage of random samples from the validation set
    if st.checkbox('Generate Image Montage'):
        st.write('Click the "Create Montage" button to refresh the montage.')
        my_data_dir = 'inputs/cherry-leaves_dataset/cherry-leaves'
        labels = os.listdir(my_data_dir + '/validation')
        label_to_display = st.selectbox(label='Choose a Label', options=labels, index=0)
        if st.button("Create Montage"):
            image_montage(dir_path=my_data_dir + '/validation',
                            label_to_display=label_to_display,
                            nrows=8, ncols=3, figsize=(10, 25))
        st.write('---')

    # Function to generate and display a montage of images from a specific label within the dataset
    def image_montage(dir_path, label_to_display, nrows, ncols, figsize=(15, 10)):
        sns.set_style("dark")
        labels = os.listdir(dir_path)

        # Check if the selected label exists in the dataset
        if label_to_display in labels:

            # Get the list of images for the selected label
            images_list = os.listdir(dir_path+ '/'+ label_to_display)
            if nrows * ncols < len(images_list):
                img_idx = random.sample(images_list, nrows * ncols)
            else:
                print(
                    f'Reduce the number of rows or columns for the montage. '
                    f'The selected label contains only {len(images)} images, '
                    f'but you requested a montage with {nrows * ncols} spaces.'
                )
                return

            # Generate indices for plotting the images
            row_indices = range(0, nrows)
            col_indices = range(0, ncols)
            plot_idx = list(itertools.product(list_rows, list_cols))

            # Create and display the montage
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
            for x in range(0, nrows * ncols):
                img = imread(dir_path + '/' + label_to_display + '/' + img_idx[x])
                img_shape = img.shape
                axes[plot_idx[x][0], plot_idx[x][1]].imshow(img)
                axes[plot_idx[x][0], plot_idx[x][1]].set_title(f"Width {img_shape[1]}px x Height {img_shape[0]}px")
                axes[plot_idx[x][0], plot_idx[x][1]].set_xticks([])
                axes[plot_idx[x][0], plot_idx[x][1]].set_yticks([])
            plt.tight_layout()

            st.pyplot(fig=fig)
            plt.show()

        else:
            print('The selected label is not available.')
            print(f'Available labels are: {labels}')








