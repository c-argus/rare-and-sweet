# Cherry Leaf Disease Detection

## Introduction

Farmy & Foods is currently facing a significant challenge with powdery mildew, a common fungal disease that affects their cherry plantations. The existing manual inspection process to detect the disease is not only time-consuming and labor-intensive but also impractical to scale across thousands of cherry trees spread over multiple farms. To address this, we propose an automated solution using machine learning, specifically a Convolutional Neural Network (CNN), to classify cherry leaf images as healthy or diseased. This approach aims to enhance efficiency, reduce costs, and maintain the quality of cherry crops. Moreover, this solution can potentially be adapted to manage diseases in other crops, further strengthening the company's agricultural management strategies.

## Objective

The goal of this project is to develop a machine learning model capable of automatically detecting powdery mildew on cherry leaves from images, replacing the current manual inspection process. By deploying a CNN-based model, we aim to quickly and accurately identify diseased leaves, thereby reducing the time and labor costs involved in manual inspections. This will help Farmy & Foods maintain high-quality standards for their cherry crops and could be expanded to other crops, improving overall disease management across the company.

## Content

* [Introduction](#introduction)
* [Objective](#objective)
* [Dataset Content](#dataset-content)
* [Business Requirements](#business-requirements)
* [Hypothesis and Validation](#hypothesis-and-validation)
  * [Hypothesis 1: Visual Differentiation](#hypothesis-1-visual-differentiation-of-healthy-and-infected-leaves)
  * [Hypothesis 2: Predicting Leaf Health Status](#hypothesis-2-predicting-leaf-health-status-using-cnn)
* [Mapping Business Requirements to Data Visualizations and ML Tasks](#mapping-business-requirements-to-data-visualizations-and-ml-tasks)
  * [Business Requirement 1](#business-requirement-1)
  * [Business Requirement 2](#business-requirement-2)
* [ML Business Case](#ml-business-case)
* [CRISP-DM Framework for Cherry Leaf Disease Detection](#crisp-dm-framework-for-cherry-leaf-disease-detection)
  * [1. Business Understanding](#1-business-understanding)
  * [2. Data Understanding](#2-data-understanding)
  * [3. Data Preparation](#3-data-preparation)
  * [4. Modeling](#4-modeling)
  * [5. Evaluation](#5-evaluation)
  * [6. Deployment](#6-deployment)
* [Dashboard Design](#dashboard-design)
  * [Summary Page](#1-summary-page)
  * [Image Visualizer](#2-image-visualizer)
  * [Powdery Mildew Detection](#3-powdery-mildew-detection)
  * [Project Hypothesis](#4-project-hypothesis)
  * [ML Performance Metrics](#5-ml-performance-metrics)
* [ML Model Justification](#ml-model-justification)
  * [Architecture](#architecture)
  * [Hyperparameter Optimization](#hyperparameter-optimization)
  * [Optimization Techniques](#optimization-techniques)
  * [Rationale for Architectural Choices](#rationale-for-architectural-choices)
  * [Model Iterations](#model-iterations)
* [Deployment](#deployment)
  * [Heroku](#heroku)
* [Main Data Analysis and Machine Learning Libraries](#main-data-analysis-and-machine-learning-libraries)
* [Additional Technologies Utilized](#additional-technologies-utilized)
* [Bugs](#unfixed-bugs)
* [Credits](#credits)
* [Acknowledgements](#acknowledgements)


## Dataset Content

- The dataset is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves).
- The dataset contains +4 thousand images taken from the client's crop fields. The images show healthy cherry leaves and cherry leaves that have powdery mildew, a fungal disease that affects many plant species. The cherry plantation crop is one of the finest products in their portfolio, and the company is concerned about supplying the market with a compromised quality product.

## Business Requirements

The cherry plantation crop from Farmy & Foods is facing a challenge where their cherry plantations have been presenting powdery mildew. Currently, the process is manual verification if a given cherry tree contains powdery mildew. An employee spends around 30 minutes in each tree, taking a few samples of tree leaves and verifying visually if the leaf tree is healthy or has powdery mildew. If there is powdery mildew, the employee applies a specific compound to kill the fungus. The time spent applying this compound is 1 minute. The company has thousands of cherry trees located on multiple farms across the country. As a result, this manual process is not scalable due to the time spent in the manual process inspection.

To save time in this process, the IT team suggested an ML system that detects instantly, using a leaf tree image, if it is healthy or has powdery mildew. A similar manual process is in place for other crops for detecting pests, and if this initiative is successful, there is a realistic chance to replicate this project for all other crops. The dataset is a collection of cherry leaf images provided by Farmy & Foods, taken from their crops.

- 1 - The client is interested in conducting a study to visually differentiate a healthy cherry leaf from one with powdery mildew.
- 2 - The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew.

## Hypothesis and Validation

### Hypothesis 1: Visual Differentiation of Healthy and Infected Leaves: Validation through EDA, Feature Extraction, and Model-Based Visualizations
Cherry leaves affected by powdery mildew can be identified by distinct visual markers that differentiate them from healthy leaves. A common indicator of this fungal infection is the presence of a thin, white or greyish powdery layer on the leaf surfaces. As the mildew spreads, these powdery patches may increase in size and density, potentially leading to leaf damage and reduced plant vitality. Advanced detection methods, including machine learning algorithms, can help in accurately distinguishing between infected and non-infected leaves.

### *Validation*: Validated through Exploratory Data Analysis (EDA), feature extraction, and model-based visualizations.
An image collage showcases the distinctive whitish spots that appear on leaves affected by a powdery mildew fungal infection. The collage includes a variety of images: some depict the average appearance of infected leaves, others display the variability among different instances of the infection, and additional images illustrate the contrasts between the average infected and healthy leaves. By carefully analyzing these images, clear patterns and differences become apparent, allowing for a more precise identification and distinction between infected and healthy foliage. This visual comparison highlights the consistent characteristics of powdery mildew, facilitating early detection and effective management of the disease.

### Hypothesis 2: Predicting Leaf Health Status Using CNN, Validation through Model Training, Evaluation Metrics, and Real-World Testing
Accurately classifying the health status of a leaf based on its visual characteristics is vital for early disease detection and effective management of plant health. This process is especially important in agriculture, where timely identification of diseases can prevent significant crop loss and ensure food security. In this study, a Convolutional Neural Network (CNN) was developed and trained to predict whether a cherry leaf is healthy or affected by powdery mildew, a common fungal disease. The CNN model was trained using a dataset of cherry leaf images, allowing it to learn and identify distinct visual patterns associated with healthy leaves and those showing signs of powdery mildew. By leveraging deep learning techniques, the model aims to provide a reliable and automated approach for monitoring plant health, potentially aiding farmers and agronomists in taking prompt action to control the spread of the disease and optimize crop yield.

### *Validation*
The CNN model was trained on a well-structured dataset of cherry leaf images, which included a balanced representation of both healthy leaves and those affected by powdery mildew. The dataset was carefully curated to include various stages of disease progression, lighting conditions, and angles to ensure robust model performance. During training, the model achieved high accuracy on both the validation and test datasets, demonstrating its ability to generalize effectively to new, unseen data.

The model's performance was rigorously evaluated using several metrics, including accuracy, loss and classification reports. These metrics provided a comprehensive understanding of the model’s ability to correctly classify both healthy and infected leaves. 
The confusion matrix, in particular, highlighted the model’s precision in distinguishing between true positives (correctly identified infected leaves) and true negatives (correctly identified healthy leaves), as well as its ability to minimize false positives and false negatives.

The high performance across these metrics indicates that the model is highly effective in distinguishing between healthy and infected leaves, thereby confirming the initial hypothesis that a CNN could be used for reliable disease detection. These results suggest that the model can serve as a valuable tool in agricultural settings for the early detection and management of powdery mildew in cherry orchards, potentially reducing the reliance on manual inspection and improving overall crop health monitoring. 

## Mapping Business Requirements to Data Visualizations and ML Tasks

### Business Requirement 1:

- As a user, I want to effortlessly navigate through an interactive dashboard to view and interpret the data presented.
- As a user, I need to see an image collage displaying either healthy or powdery mildew-infected cherry leaves, allowing me to visually distinguish between them.
- As a user, I want to view and toggle between visual graphs showing average images (and differences between averages) as well as image variability for both healthy and mildew-affected cherry leaves, helping me to recognize the visual markers indicative of leaf quality.

### Business Requirement 2:

- As a user, I want to utilize a machine learning model to receive a classification prediction for a provided image of a cherry leaf.
- As a user, I need the ability to input new raw data on a cherry leaf, clean it, and then run it through the provided model.
- As a user, I want to upload the cleaned data to the dashboard so the model can process it and immediately determine if the cherry leaf is healthy or infected with powdery mildew.
- As a user, I want to save the model's predictions in a CSV file with timestamps so that I can maintain a record of all predictions made.


## ML Business Case

* The objective is to create a machine learning tool that can effectively and precisely identify whether a cherry leaf is healthy or affected by powdery mildew, thereby improving the efficiency of inspections and enhancing labor quality.

* A dataset provided by the client will be utilized to train this ML tool, which aims to accurately distinguish between healthy leaves and those infected with mildew.

* The client also requires a user-friendly interface that facilitates the quick uploading of leaf images and achieves at least 97% accuracy in determining the health status of each leaf.

* To maintain the confidentiality of proprietary information, suitable measures will be taken to secure customer data.

* The success of this ML tool will be evaluated based on its accuracy and efficiency in detecting infected leaves, as well as its capacity to decrease the time and costs linked to manual inspections.

* Additionally, the ML tool could be expanded to assess the success of powdery mildew treatments by determining whether the treated leaves have returned to a healthy state.

* There is also potential for the ML tool to be adapted for other crops, particularly those requiring pest detection, to further enhance the efficiency and precision of inspection processes.

* The performance of the ML tool will be validated by training and testing the model on a dataset of cherry leaf images labeled as either healthy or infected with powdery mildew, and its effectiveness will be measured using relevant metrics like accuracy or the F1 score.

* The criteria for the model's success include:

  * Demonstrating the ability to visually differentiate between healthy cherry leaves and those with powdery mildew.
  * Accurately predicting whether a cherry leaf is healthy or infected.
  * Achieving over 90% accuracy on test data.


## CRISP-DM Framework for Cherry Leaf Disease Detection
To effectively guide the machine learning process, we adopt the **Cross-Industry Standard Process for Data Mining (CRISP-DM)** framework, which provides a structured approach across six phases: Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, and Deployment.

### 1. Business Understanding
**Objective:** Align machine learning efforts with Farmy & Foods' goal of automating disease detection to save time and reduce costs associated with manual inspections.

**Project Background:**
Farmy & Foods currently employs a manual process to detect powdery mildew, which is not scalable due to the large number of cherry trees.
An automated solution using machine learning is required to streamline this process.

**Business Objectives:**
- Rapid and accurate detection of powdery mildew in cherry leaves to reduce labor costs and prevent crop quality degradation.

**Success Criteria:**
- A machine learning model with at least 97% accuracy in identifying healthy and diseased leaves.
- The model should be deployed in an accessible format for real-time use by field workers.

### 2. Data Understanding
**Objective:** Acquire and explore the dataset to understand the nature of the data and any quality issues.

**Data Collection:**
- The dataset consists of over 4,000 cherry leaf images sourced from Farmy & Foods' crop fields.
- Images are divided into two categories: healthy and those affected by powdery mildew.

**Data Exploration:**
- Perform initial analysis with statistical summaries and visualizations.
- Understand the distribution of classes and detect any anomalies or data quality issues.

**Data Quality Verification:**
- Check for missing or mislabeled images, inconsistent dimensions, or image quality issues that could affect model training.

### 3. Data Preparation
**Objective:** Prepare a clean and structured dataset suitable for modeling.

**Data Cleaning:**
- Remove any corrupted images or those that do not meet quality standards required for accurate modeling.

**Data Transformation:**
- Resize images to a uniform dimension.
- Normalize pixel values.
- Perform data augmentation techniques (e.g., rotation, flipping) to improve model robustness.

**Data Splitting:**
- Split the dataset into training, validation, and test sets to ensure reliable model evaluation and prevent overfitting.

### 4. Modeling
**Objective:** Develop machine learning models to predict the health status of cherry leaves.

**Model Selection:**
- Choose a Convolutional Neural Network (CNN) architecture suitable for image classification tasks.

**Model Training:**
- Train the CNN using the prepared dataset.
- Experiment with different hyperparameters (e.g., learning rate, batch size, number of epochs) to optimize performance.

**Model Testing:**
- Validate model performance using the test set to evaluate its generalization capabilities.

### 5. Evaluation
**Objective:** Assess the model's performance against business and technical objectives.

**Model Evaluation Metrics:**
- Evaluate the model using metrics such as accuracy, precision, recall, F1-score, and confusion matrix to ensure it meets the required performance standards.

**Business Relevance:**
- Confirm that the model can achieve the desired accuracy and efficiency to be a viable replacement for manual inspection processes.

**Iterative Refinement:**
- Based on evaluation outcomes, refine the model by revisiting the data preparation or modeling steps to enhance performance.

### 6. Deployment
**Objective:** Implement the model in a real-world setting for practical use by Farmy & Foods.

**Deployment Strategy:**
- Deploy the trained model on a cloud platform like Heroku to allow for real-time predictions accessible through a user-friendly interface.

**Integration with Business Processes:**
- Ensure the deployment pipeline aligns with existing workflows for disease management, allowing seamless integration for field workers.

**Monitoring and Maintenance:**
- Establish monitoring processes to track model performance over time.
- Retrain the model periodically with new data to maintain accuracy.

## Dashboard Design

The dashboard provides the following features:

- **Upload Image**: A widget to upload a leaf image for prediction.

- **Prediction Result**: A section displaying the model's prediction (healthy or powdery mildew).

- **Visualizations**: Interactive charts and graphs showing the distribution of healthy and diseased leaves, model accuracy, and other relevant metrics.

- **Feedback Section**: A form allowing users to provide feedback on the model's performance.

- **Navigation Menu**: Buttons to navigate between different sections of the dashboard.

**1. Summary Page**
The Project Summary Page offers crucial information about the project's background, such as its origin and the customer who commissioned it. It outlines the business requirements set forth by the customer, which establish the criteria for the project's success. Furthermore, the page details the project's goals and the methods to be employed in reaching them.

![DashboardSummary](assets/DasboardSummary.png)

**2. Image Visualizer**
The ImageVisualizer page addresses the project's initial Data Analysis business goal. It includes plots that can be conveniently displayed or hidden using the built-in toolbar.
This app page also features a tool for creating image montages, allowing the user to choose a label class to display a montage. This montage is created using a graphical presentation of random images from the validation set.

![DashboardImageVisualizer](assets/DashboardImageVisualizer00.png)
![DashboardImageVisualizer](assets/DashboardImageVisualizer01.png)
![DashboardImageVisualizer](assets/DashboardImageVisualizer02.png)
![DashboardImageVisualizer](assets/DashboardImageVisualizer05.png)
![DashboardImageVisualizer](assets/DashboardImageVisualizer03.png)
![DashboardImageVisualizer](assets/DashboardImageVisualizer04.png)

**3. Powdery Mildew Detection**
The Powdery Mildew Detection tool offers a downloadable dataset of cherry leaf images, both infected and uninfected, for live predictions on Kaggle. The user interface features a file uploader that enables users to upload multiple cherry leaf images. Upon uploading, the system displays the image, a bar chart of the predicted result, and a statement indicating whether the leaf is infected with powdery mildew, along with the probability score. Additionally, a table lists the image names and corresponding prediction results. There is also a download button to save the report as a .csv file, and a link to the Readme.md file for more information about the project.

![DashboardPowderyMildewDetection](assets/DashboardPMD00.png)
![DashboardPowderyMildewDetection](assets/DashboardPMD01.png)
![DashboardPowderyMildewDetection](assets/DashboardPMD02.png)
![DashboardPowderyMildewDetection](assets/DashboardPMD03.png)
![DashboardPowderyMildewDetection](assets/DashboardPMD04.png)
![DashboardPowderyMildewDetection](assets/DashboardPMD05.png)
![DashboardPowderyMildewDetection](assets/DashboardPMD06.png)

**4. Project Hypothesis**
The Hypothesis page presents the two hypotheses and the project's outcome goals, along with the success metrics.

![DashboardHypothesis](assets/DashboardHypothesis.png)

**5. ML Performance Metrics**
The ML Prediction Metrics page provides a summary of the machine learning model's performance in predicting whether cherry leaves are infected with powdery mildew. It includes visualizations of label distribution across training, validation, and test sets, and outlines the model’s accuracy and loss during training, demonstrating effective learning and minimal overfitting. The page also presents the model's high accuracy and low loss on the test set, indicating strong generalization to new data. Additionally, key metrics such as loss and accuracy are explained, providing insights into their significance in evaluating and optimizing the model’s performance.

![DashboardMLPerfomanceMetrics](assets/DashboardMLPM00.png)
![DashboardMLPerfomanceMetrics](assets/DashboardMLPM01.png)
![DashboardMLPerfomanceMetrics](assets/DashboardMLPM02.png)

## ML Model Justification
The goal of the development was to build a robust and accurate model specifically tailored to predict whether a cherry leaf is healthy or affected by powdery mildew. The model is designed to effectively handle image data and perform binary classification with high accuracy.

### Architecture
The model architecture is a Convolutional Neural Network (CNN) designed to capture spatial hierarchies in the input images. The CNN consists of three convolutional layers, each followed by a max pooling layer, a dense layer with a ReLU activation function, and a final output layer with a sigmoid activation function. The CNN structure is optimized for extracting features from images by using convolutional filters to detect patterns and pooling layers to down-sample the feature maps. 

The dense layer aggregates these features and maps them to a binary output using the ReLU activation function, while the final output layer employs a sigmoid activation function to output a value between 0 and 1, representing the probability of the input image belonging to one of the two classes (healthy or powdery mildew).

The number of filters in the convolutional layers is carefully chosen to capture increasing levels of feature complexity:
- The first convolutional layer has 32 filters.
- The second convolutional layer has 64 filters.
- The third convolutional layer also has 64 filters.

This progression in the number of filters allows the model to learn both low-level features in the earlier layers and more abstract, high-level features in the later layers, which is essential for accurate image classification.

The decision to use 32, 64, and 64 filters is based on common practices for image classification tasks, especially when the input images have a moderate level of complexity and size. This configuration allows the model to effectively learn the spatial hierarchies present in the cherry leaf images.

### Hyperparameter Optimization
The model's hyperparameters were carefully selected based on empirical experimentation and common practices for CNNs:
- **Number of Units in Dense Layer**: The dense layer contains 128 units. This number was chosen to balance model complexity with the available data, allowing the model to learn useful representations without overfitting.
- **Learning Rate of Optimizer**: The Adam optimizer is used with a learning rate selected based on performance on a validation set. This learning rate was chosen to ensure stable convergence during training.

**Early Stopping** is used to prevent overfitting by monitoring the validation loss and stopping training when the loss ceases to decrease. This technique ensures that the model does not learn to fit noise in the training data, thereby improving generalization to new, unseen data.

### Optimization Techniques
**Dropout** is employed as a regularization method to prevent overfitting. A dropout layer with a rate of 0.5 is added after the dense layer. This technique randomly sets a fraction of input units to zero during training, which helps prevent the network from becoming too dependent on any one neuron, encouraging more robust feature learning.

**Adam Optimizer** is chosen for its ability to efficiently minimize the binary cross-entropy loss function. Adam combines the benefits of the AdaGrad and RMSProp algorithms to adapt the learning rate for each parameter, which makes it well-suited for handling sparse gradients and improving the training process for deep networks.

**Data Augmentation** is applied to increase the diversity of the training data without actually collecting new data. Techniques such as rotation, width and height shifts, shear, zoom, and horizontal and vertical flips are applied. This improves the model's ability to generalize by exposing it to a wider variety of data during training.

**Binary Cross-Entropy Loss Function** is used to optimize the model parameters, as this loss function is well-suited for binary classification tasks where the model outputs a probability value between 0 and 1.

### Rationale for Architectural Choices
Using three convolutional layers with increasing numbers of filters is a strategic choice to capture various levels of feature complexity. The initial layers focus on detecting edges and textures, while the later layers are tuned to recognize more complex shapes and patterns, which is critical for distinguishing between healthy and diseased leaves.

**Sigmoid Activation Function** is chosen for the output layer because it effectively maps the model's outputs to a probability range between 0 and 1, which is suitable for binary classification tasks. This allows the model to produce a clear probabilistic output that can be easily thresholded to make binary decisions.

The **ReLU Activation Function** in the hidden dense layer is used to introduce non-linearity into the model, enabling it to learn complex patterns and interactions between the features extracted by the convolutional layers.

**Early Stopping** and **Dropout** are specifically used to prevent overfitting, ensuring that the model remains generalizable and performs well on unseen data, which is crucial given the limited dataset size.

By balancing the model complexity with the amount of data available and incorporating various regularization and optimization techniques, the CNN is both effective in capturing relevant features for the task and robust against overfitting, making it a suitable choice for the binary classification of cherry leaf health.

### Model Iterations
The model demonstrates excellent training and validation performance, with accuracy consistently high and loss significantly reduced over multiple epochs, as shown in the line charts. The final performance metrics reflect these trends, with the model achieving an accuracy of 99.88% and a very low loss of 0.63%, indicating strong model convergence and predictive capabilities. Despite these impressive metrics, there may still be challenges with specific aspects of performance, such as recall, particularly if the model has difficulty correctly identifying all instances of certain classes (e.g., infected leaves misidentified as healthy), suggesting further tuning might be needed to address class-specific errors.

![ModelGraph](assets/ModelIterations.png)

![LossAccuracy](assets/LossAccuracy.png)

## Deployment

### Heroku

- The App live is: [Cherry Leaves Disease Detector](https://mildewdetectionincherryleaf-eb586cae0ee9.herokuapp.com/)
- Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
- The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click the button Open App on the top of the page to access your App.
6. If the slug size is too large, then add large files not required for the app to the .slugignore file.

## Main Data Analysis and Machine Learning Libraries

- [NumPy](https://numpy.org): Utilized for data processing, preparation, and visualization. It serves as the foundation for TensorFlow.

- [Pandas](https://pandas.pydata.org): Facilitates the conversion of numerical data into DataFrames for easier manipulation and analysis.

- [Matplotlib](https://matplotlib.org): Used for reading, processing, and visualizing image data, as well as generating graphs from tabular data.

- [Seaborn](https://seaborn.pydata.org): Aids in data visualization and presentation, including creating confusion matrix heatmaps and scatter plots for image dimensions.

- [Plotly](https://plotly.com): Enables graphical visualization of data through interactive charts and plots.

- [TensorFlow](https://www.tensorflow.org): A powerful machine learning library employed for building models.

- [Keras Tuner](https://keras.io/keras_tuner): Helps tune hyperparameters to identify the best combinations for optimal model accuracy.

- [Scikit-learn](https://scikit-learn.org): Provides tools for calculating class weights to address target imbalance and for generating classification reports.

### Additional Technologies Utilized

- [Streamlit](https://streamlit.io): A library designed for creating interactive web applications and dashboards tailored for data science projects.

- [Heroku](https://www.heroku.com): Platform used for deploying the dashboard as a web application.

- [Git/GitHub](https://github.com): Tools for version control and managing source code.

- [VSCode](https://code.visualstudio.com): An integrated development environment (IDE) used for local coding and development.

## Bugs
* An AttributeError occurred in the Cherry Leaves Disease Detector application once deployed on Heroku. 
The error, AttributeError: 'DataFrame' object has no attribute 'append', was due to the deprecation of the .append() method in Pandas version 1.4.0, which was previously used for appending rows to a DataFrame. 
The error was in the page_cherry_leaves_detector_body function in page_cherry_leaves_detector.py. and in the page_cherry_leaves_detector_body function within page_cherry_leaves_detector.py
To fix this, replaced .append() with a list to collect data and convert the list to a DataFrame once using pd.DataFrame(). 
This approach improves performance, memory efficiency, and compatibility with the latest Pandas version. 
The modified code initializes an empty list, collects image data and prediction results into this list, and then creates the DataFrame after all data is collected. 
![Bug1](assets/Bug1.png)
![Bug2](assets/Bug2.png)


## Credits

### Content
- The Malaria Walkthrough Project from Code Institute was used as an educational tool and provided guidance throughout the development of this project.
- The text for the Home page was adapted based on the content from this website: [Metos - Cherry Crop Disease Models](https://metos.ca/software-tools/crop-disease-models/fruits-horticulture/cherry/).
- The [Streamlit documentation](https://docs.streamlit.io/) was referenced for further understanding and troubleshooting.
- The [Huddy2022/milestone-project-mildew-detection-in-cherry-leaves](https://github.com/Huddy2022/milestone-project-mildew-detection-in-cherry-leaves) was consulted as a reference.
The [DenysRudenko/Project5_Mildew-detection-cherry-leaves](https://github.com/DenysRudenko/Project5_Mildew-detection-cherry-leaves)

### Media
- The app icon is taken from [Emojipedia](https://emojipedia.org/).

## Acknowledgements
* Thanks to my mentor Precious Ijege for the support.
* Thanks to my fiance and friends that helped me in the hard moments throughout the project.
