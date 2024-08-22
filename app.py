import streamlit as st
from app_pages.multipage import MultiPage


# Load individual page scripts
from app_pages.page_summary import page_summary_body
from app_pages.page_image_visualizer import page_image_visualizer_body
from app_pages.page_cherry_leaves_detector import page_cherry_leaves_detector_body
from app_pages.page_project_hypothesis import page_project_hypothesis_body
from app_pages.page_ml_performance import page_ml_performance_body

# Create an instance of the app
app = MultiPage(app_name="Cherry Leaves Disease Detector")

# Add app pages
app.add_page("Project Summary", page_summary_body)
app.add_page("Image Visualizer", page_image_visualizer_body)
# app.add_page("Powdery Mildew Detection", page_cherry_leaves_detector_body)
# app.add_page("Project Hypothesis", project_hypothesis_body)
# app.add_page("ML Perfomance Metrics", page_ml_performance_body)

# Run the app
app.run()
