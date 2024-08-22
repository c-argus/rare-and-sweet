import streamlit as st
import matplotlib.pyplot as plt

# Function to display the Project Insights and Findings page
def page_project_hypothesis_body():
    st.title("Project Hypothesis and Validation")

    st.write("## Hypotesis 1 and Validation")
    st.write("* Visual Differentiation of Healthy and Infected Leaves: Validation through EDA, Feature Extraction, and Model-Based Visualizations")

    st.info(
        f'Plants suffering from a fungal infection, particularly powdery mildew, '
        f'display unique visual traits on their leaves, ' 
        f'including pale-grey or white powdery spots.'
    )

    # Displaying a success message with the validation of the observation
    st.success(
        f'### Validation:\n\n'
        f'An image collage displays the typical whitish spots on leaves impacted '
        f'by a powdery mildew fungal infection. '
        f'By examining the average images, variability images, and the contrasts between these averages, '
        f'clear patterns become evident, aiding in the distinction between infected and healthy leaves.'
    )

    st.write("**For a more detailed view, please navigate to the visualizer tab.**")

    st.write("## Hypotesis 2 and Validation")
    st.write("* Predicting Leaf Health Status Using CNN: Validation through Model Training, Evaluation Metrics, and Real-World Testing")

    st.info(
        f'The ability to accurately classify the health status of a leaf based on its visual characteristics '
        f'is crucial for early disease detection. A Convolutional Neural Network (CNN) was trained to predict '
        f'whether a cherry leaf is healthy or affected by powdery mildew.'
    )

    st.success(
        f'### Validation:\n\n'
        f'The CNN model was trained on a well-structured dataset of cherry leaves, achieving high accuracy on both '
        f'the validation and test datasets. The model’s performance was evaluated using various metrics, including '
        f'accuracy, loss, confusion matrices, and classification reports. The results indicate that the model is highly '
        f'effective in distinguishing between healthy and infected leaves, confirming the hypothesis.'
    )

    st.write("**For detailed metrics and model evaluation, please navigate to the ML Perfomance Metrics tab.**")

