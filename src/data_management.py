import pandas as pd
import streamlit as st
import base64
import joblib


def download_dataframe_as_csv(df: pd.DataFrame, filename="data.csv"):
    """
    Generates a link to download the given DataFrame as a CSV file.
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV file</a>'
    return href

def load_pkl_file(file_path: str):
    """
    Load a pickle file from the given path.
    """
    return joblib.load(file_path)