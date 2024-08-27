# Cherry Leaf Disease Detection

## Introduction

Farmy & Foods is currently facing a significant challenge with powdery mildew, a common fungal disease that affects their cherry plantations. The existing manual inspection process to detect the disease is not only time-consuming and labor-intensive but also impractical to scale across thousands of cherry trees spread over multiple farms. To address this, we propose an automated solution using machine learning, specifically a Convolutional Neural Network (CNN), to classify cherry leaf images as healthy or diseased. This approach aims to enhance efficiency, reduce costs, and maintain the quality of cherry crops. Moreover, this solution can potentially be adapted to manage diseases in other crops, further strengthening the company's agricultural management strategies.

## Objective

The goal of this project is to develop a machine learning model capable of automatically detecting powdery mildew on cherry leaves from images, replacing the current manual inspection process. By deploying a CNN-based model, we aim to quickly and accurately identify diseased leaves, thereby reducing the time and labor costs involved in manual inspections. This will help Farmy & Foods maintain high-quality standards for their cherry crops and could be expanded to other crops, improving overall disease management across the company.

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

### Validation: Validated through Exploratory Data Analysis (EDA), feature extraction, and model-based visualizations.
An image collage showcases the distinctive whitish spots that appear on leaves affected by a powdery mildew fungal infection. The collage includes a variety of images: some depict the average appearance of infected leaves, others display the variability among different instances of the infection, and additional images illustrate the contrasts between the average infected and healthy leaves. By carefully analyzing these images, clear patterns and differences become apparent, allowing for a more precise identification and distinction between infected and healthy foliage. This visual comparison highlights the consistent characteristics of powdery mildew, facilitating early detection and effective management of the disease.

### Hypothesis 2: Predicting Leaf Health Status Using CNN, Validation through Model Training, Evaluation Metrics, and Real-World Testing
Accurately classifying the health status of a leaf based on its visual characteristics is vital for early disease detection and effective management of plant health. This process is especially important in agriculture, where timely identification of diseases can prevent significant crop loss and ensure food security. In this study, a Convolutional Neural Network (CNN) was developed and trained to predict whether a cherry leaf is healthy or affected by powdery mildew, a common fungal disease. The CNN model was trained using a dataset of cherry leaf images, allowing it to learn and identify distinct visual patterns associated with healthy leaves and those showing signs of powdery mildew. By leveraging deep learning techniques, the model aims to provide a reliable and automated approach for monitoring plant health, potentially aiding farmers and agronomists in taking prompt action to control the spread of the disease and optimize crop yield.

### Validation
The CNN model was trained on a well-structured dataset of cherry leaf images, which included a balanced representation of both healthy leaves and those affected by powdery mildew. The dataset was carefully curated to include various stages of disease progression, lighting conditions, and angles to ensure robust model performance. During training, the model achieved high accuracy on both the validation and test datasets, demonstrating its ability to generalize effectively to new, unseen data.

The model's performance was rigorously evaluated using several metrics, including accuracy, loss and classification reports. These metrics provided a comprehensive understanding of the model’s ability to correctly classify both healthy and infected leaves. 
The confusion matrix, in particular, highlighted the model’s precision in distinguishing between true positives (correctly identified infected leaves) and true negatives (correctly identified healthy leaves), as well as its ability to minimize false positives and false negatives.

The high performance across these metrics indicates that the model is highly effective in distinguishing between healthy and infected leaves, thereby confirming the initial hypothesis that a CNN could be used for reliable disease detection. These results suggest that the model can serve as a valuable tool in agricultural settings for the early detection and management of powdery mildew in cherry orchards, potentially reducing the reliance on manual inspection and improving overall crop health monitoring. 

## The rationale to map the business requirements to the Data Visualisations and ML tasks

- List your business requirements and a rationale to map them to the Data Visualisations and ML tasks.

## ML Business Case

To meet the business requirements, the project aims to develop a predictive model using machine learning. The primary ML task is binary image classification using a CNN. This model will automate the process of detecting powdery mildew, significantly reducing inspection times and labor costs, and potentially be scalable to other crops and diseases.

### Machine Learning Workflow

1. **Data Collection**: Gather images from the Kaggle dataset provided by Farmy & Foods.
2. **Data Preprocessing**: Resize images, normalize pixel values, and split into training and test sets.
3. **Model Building**: Develop a CNN using a framework such as TensorFlow or PyTorch.
4. **Model Training**: Train the CNN on the training set and validate its performance using a test set.
5. **Evaluation**: Use metrics like accuracy, precision, recall, and F1-score to evaluate the model's performance.
6. **Deployment**: Deploy the model on Heroku for real-time predictions.

## Dashboard Design

The dashboard will provide the following features:
- **Upload Image**: A widget to upload a leaf image for prediction.
- **Prediction Result**: A section displaying the model's prediction (healthy or powdery mildew).
- **Visualizations**: Interactive charts and graphs showing the distribution of healthy and diseased leaves, model accuracy, and other relevant metrics.
- **Feedback Section**: A form allowing users to provide feedback on the model's performance.
- **Navigation Menu**: Buttons to navigate between different sections of the dashboard.

## Unfixed Bugs

At the time of deployment, some bugs related to the image upload functionality and real-time prediction display remain unfixed. These issues stem from limitations in the library used for image processing and the asynchronous handling of requests in the web framework. Future updates will focus on addressing these bugs as more advanced libraries and solutions become available..

## Deployment

### Heroku

- The App live link is: `https://YOUR_APP_NAME.herokuapp.com/`
- Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
- The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click the button Open App on the top of the page to access your App.
6. If the slug size is too large, then add large files not required for the app to the .slugignore file.

## Main Data Analysis and Machine Learning Libraries

The following libraries were used in the project:
- **TensorFlow/Keras**: Used for building and training the Convolutional Neural Network (CNN).
  - Example: `model = keras.Sequential([...])`
- **OpenCV**: Used for image processing and augmentation.
  - Example: `img = cv2.imread(image_path)`
- **Matplotlib and Seaborn**: Used for creating visualizations and plots.
  - Example: `plt.imshow(image)`
- **NumPy and Pandas**: Used for data manipulation and analysis.
  - Example: `data = pd.read_csv('data.csv')`

## Credits

### Content

- The text for the Home page was taken from Wikipedia Article A.
- Instructions on how to implement form validation on the Sign-Up page were taken from [Specific YouTube Tutorial](https://www.youtube.com/).
- The icons in the footer were taken from [Font Awesome](https://fontawesome.com/).

### Media

- The photos used on the home and sign-up page are from This Open-Source site.
- The images used for the gallery page were taken from this other open-source site.

## Acknowledgements (optional)

- Thank the people who provided support throughout this project.
