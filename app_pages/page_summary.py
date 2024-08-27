import streamlit as st

# Function to display the 'Quick Project Summary' section on the page 
def page_summary_body():

    # Displaying the project title and introduction
    st.title('Project Summary')
    st.write('### Introduction ')
    st.write('#### Powdery Mildew Detector on Cherry Trees')

    st.write(
        f'Powdery Mildew is a widespread fungal infection that impacts a variety of plants, '
        f'visibly identified by light grey or white powdery spots that primarily appear on leaves, '
        f'but can also affect stems, flowers, and fruits. '
        f'As the disease progresses, these spots '
        f'can expand, covering large portions of the leaves, particularly on newer growth.\n\n'

        f'While the disease is seldom fatal, if left untreated, powdery mildew can cause significant '
        f'damage by robbing the plant of essential water and nutrients. Common symptoms include yellowing, '
        f'withering, leaf distortion, reduced growth, fewer flowers, and delayed development.\n\n'

        f'The current method of manually inspecting each cherry tree is time-consuming, taking roughly '
        f'30 minutes per tree, plus an additional minute for treatment if required. Given the vast number of '
        f'trees spread across various farms, this approach is not scalable. To overcome this challenge, this project '
        f'has proposed the implementation of a machine learning (ML) system that can quickly detect powdery mildew in '
        f'images of cherry trees.\n\n'
    )

    # Provide links for additional information and references
    st.info(
        f'For further information:\n'
        f'* Refer to the [Project README file](https://github.com/c-argus/rare-and-sweet/blob/main/README.md).\n'
    )

    # Summarizing the project's business objectives
    st.write(
        f'### Objectives:\n\n'
        f'The project focuses on two key business:\n'
        f'* 1- To visually differentiate a healthy cherry leaf from one with powdery mildew.\n' 
        f'* 2- To predict whether a cherry tree is healthy or suffering from powdery mildew.' 
    )

    # Briefly describing the project dataset
    st.write('### Project Dataset:')
    st.write(
        f'The dataset, consisting of cherry tree leaf images provided by Farmy & Foods, '
        f'is the foundation for training and evaluating the model.\n'
        f'This dataset is available on [Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves) and '
        f'includes over 4,000 images, with a selected subset used for efficient model training.'
    )