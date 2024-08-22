import streamlit as st
import matplotlib.pyplot as plt

# Function to display the Project Insights and Findings page
def page_project_hypothesis_body():
    st.title("Project Hypothesis and Validation")

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


