import streamlit as st 
import matplotlib.pyplot as plt
import pandas as pd 
from matplotlib.image import imread
from src.machine_learning.evaluate_clf import load_test_evaluation

def page_ml_performance():
    version = 'v1'
    '''
    Function that displays the performance metrics of the
    machine learning model v1
    '''

    st.title("Train, Validation and Test Set: Labels Frequencies")
    st. write(
        f'Dataset was divided into 3 subsets: \n\n'
        f'* Training Set: comprises 70% of the data.\n'
        f'* Test Set: comprises 10% of the data.\n'
        f'* Validation Set: comprises 20% of the data. \n'
    )
    st.write(
        f"* The image dimensions for the ML model are based on the average size of all images in the 'Trainset':\n\n"
        f'(256, 256, 3)'
    )
    st.write(
        f"Labels were categorized as: '_healthy_' and '_powdery_mildew_'."
    )

    labels_distribution = plt.imread(f'outputs/{version}/labels_distribution.png')
    st.image(labels_distribution, caption='Labels Proportion on Train, Validation and Test Sets')
    st.write("---")

    st.write("### Model History")
    col1, col2 = st.beta_columns(2)
    with col1: 
        model_acc = plt.imread(f"outputs/{version}/model_training_acc.png")
        st.image(model_acc, caption='Model Training Accuracy')
    with col2:
        model_loss = plt.imread(f"outputs/{version}/model_training_losses.png")
        st.image(model_loss, caption='Model Training Losses')
    st.write("---")

    st.write("### Generalised Performance on Test Set")
    st.dataframe(pd.DataFrame(load_test_evaluation(version), index=['Loss', 'Accuracy']))
