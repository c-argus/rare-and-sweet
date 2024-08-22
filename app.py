import streamlit as st
from app_pages.multipage import MultiPage


# Load individual page scripts
from app_pages.page_summary import page_summary_body
from app_pages.page_image_visualizer import page_image_visualizer_body
from app_pages.page_cherry_leaves_detector import page_cherry_leaves_detector
from app_pages.page_model_training import page_model_training_body
# from app_pages.page_model_evaluation import page_model_evaluation_body

# Create an instance of the app
app = MultiPage(app_name="Cherry Leaves Disease Detector")

# Add app pages
app.add_page("Project Summary", page_summary_body)
app.add_page("Image Visualizer", page_image_visualizer_body)
# app.add_page("Image Analysis", page_image_analysis_body)
# app.add_page("Model Training", page_model_training_body)
# app.add_page("Model Evaluation", page_model_evaluation_body)

# Run the app
app.run()
