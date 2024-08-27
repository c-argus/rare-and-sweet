import streamlit as st 
import matplotlib.pyplot as plt
import pandas as pd 
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
        f'(110, 110, 3)'
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

    st.info(
        f'These results indicate that the model has attained an extremely low loss and very high accuracy, which reflects outstanding performance. '
        f"A loss value of 0.0065 suggests that the model's predictions are highly accurate on average, while an accuracy of "
        f'0.9988 indicates that the model correctly classifies the data points in approximately 99.8% of cases.'
    )

    st.write("### Key Concepts in Performance Metrics:")
    st.write(
        f"**Loss and Accuracy** are two fundamental metrics used to assess the effectiveness of machine learning models.\n\n"
    
        f'* **Loss:**\n\n'
        f'  - Loss quantifies the errors made by a model, serving as an indicator of its performance.\n\n'
        f'  - A **high loss** suggests that the model is making considerable errors, while a **low loss** indicates better performance with fewer errors.\n\n'
        f'  - Tracking loss over time provides insights into the model\'s learning progress:\n\n'
        f'    - **Fluctuating loss** might signal that the model is struggling to learn effectively.\n\n'
        f'    - If the loss decreases during training but remains high on validation data, this could be a sign of **overfitting**, where the model is not generalizing well to new data.\n\n'
    
        f'* **Accuracy:**\n\n'
        f'  - Accuracy measures the proportion of correct predictions made by the model relative to the total predictions.\n\n'
        f'  - A **high accuracy** reflects that the model is making correct predictions most of the time, while **low accuracy** indicates frequent incorrect predictions.'
    )

    st.write(
        f'**Learning Curves:**\n\n'
        f'  - A learning curve is a visual representation that shows the evolution of a particular metric during the model’s training process.\n\n'
        f'  - Typically, the **x-axis** represents time or training iterations, and the **y-axis** shows the metric being tracked, such as error rate or accuracy.\n\n'
        f'  - Learning curves are valuable tools for monitoring how well the model is learning, diagnosing potential issues, and optimizing its performance.\n\n'
    
        f'* **Examples of Learning Curves:**\n\n'
        f'  - Common types of learning curves include those showing changes in **loss**, **accuracy**, **precision**, and **recall** over time.\n\n'
        f'  - Improvement in these metrics over time generally indicates that the model is learning and improving.\n\n'
        f'  - However, if a learning curve **plateaus**, it may suggest that the model has reached its learning capacity, and further training may not yield significant improvements.'
    )

    st.write(
        f'**Analyzing Model Behavior:**\n\n'
        f'  - By observing learning curves, we can detect issues in the model’s training process.\n\n'
        f'  - **Sudden shifts** or **lack of progress** in the curves might indicate underlying problems like **overfitting** (model fits the training data too closely) or **underfitting** (model fails to capture the underlying patterns in the data).\n\n'
        f'  - **Consistent monitoring** of these curves helps ensure the model is on the right track and performing optimally.'
    )

    st.write(
        f'### Overall:\n\n'
        f'Learning curves provide crucial insights into the model’s training dynamics, enabling better evaluation and optimization of its performance. '
        f'By carefully interpreting these curves, we can make informed decisions to improve model accuracy and reduce errors.'
    )

    



