import base64
from datetime import datetime 
import joblib


def download_dataframe_as_csv(df):
    """
    Generates a link to download the given DataFrame as a CSV file.
    """
    daytime_now = datetime.now().strftime("%d%b%Y_%Hh%Mmin%Ss")
    csv = df.to_csv().encode()
    b64 = base64.b64encode(csv).decode()
    href = (
        f' <a href="data:file/csv;base64,{b64}" download="Report {daytime_now}.csv" '
        f'target="_blank">Download Report </a>'
    )
    return href

def load_pkl_file(file_path):
    return joblib.load(filename=file_path)


    

