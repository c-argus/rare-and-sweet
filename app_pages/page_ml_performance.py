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
        f'* Training Set: consisting of 70% of the data.\n'
        f'* Test Set: consisting of 10% of the data.\n'
        f'* Validation Set: consisting of 20% of the data. \n'
    )
    st.write(
        f"* The image dimensions for the ML model are determined by the average size of all images in the Training Set: \n\n"
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

    st.write('### Overview of Performance Metrics')
    st.info(
        f"Following an extensive evaluation of the model's performance on both the training and validation datasets, it can be confidently stated "
        f'that the CNN model developed for classifying cherry leaves is performing at a high level of accuracy, as expected.\n\n'
        f'*Overfitting has been kept to a minimum,* which indicates that the model is likely to generalize effectively to new, unseen data.\n\n'
        f'The model was trained over several epochs (training cycles), during which it achieved low loss values and high accuracy. '
        f'The accuracy metrics demonstrate a consistent upward trajectory, reflecting the model’s improved ability to differentiate between healthy leaves and those affected by powdery mildew over time. '
        f'Although the accuracy lines for training and validation sets are not exactly aligned, they remain closely associated throughout the training process, with only minor deviations.\n\n'
        f'Additionally, the loss metrics show a steady decline, indicating that the model is progressively reducing errors. '
        f'The loss lines for both the training and validation datasets closely follow one another, especially in the later stages of training, where they converge, suggesting minimal differences between the datasets.\n\n'
        f'Overall, the accuracy and loss trends demonstrate that the model is learning effectively, with accuracy improving and loss decreasing, confirming that the model is robust and capable of performing well on unseen data.'
    )

    st.write("### Generalised Performance on Test Set")
    st.dataframe(pd.DataFrame(load_test_evaluation(version), index=['Loss', 'Accuracy']))
