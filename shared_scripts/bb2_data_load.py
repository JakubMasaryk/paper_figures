import importlib.util
if importlib.util.find_spec("b2sdk") is None:
    %pip install b2sdk
import b2sdk
from b2sdk.v2 import InMemoryAccountInfo, B2Api
import numpy as np
import pandas as pd
import cv2 as cv
import os
from io import BytesIO
from pathlib import Path


#BackBlaze B2 bucket name (do NOT modify)
data_bucket_name= 'figure-datasets'


#BackBlaze B2 authentication (do NOT modify)
bucket_key_id= '003b5f880f95dd4000000000b'
bucket_key= 'K003DqMjc6XUHZ3rfE9j+lFyuzk1WwE'


# load selected dataset into a pd data-frame
def BB2_load_into_dataframe(file_name,  # file name with extension (.csv or .xlsx)
                            bb_bucket_name=data_bucket_name,
                            bb_key_id=bucket_key_id,
                            bb_key=bucket_key):
    # check the file format
    extension = Path(file_name).suffix.lower()
    if extension not in [".csv", ".xlsx"]:
        raise ValueError(f"unsupported file type: '{extension}'. Only CSV and XLSX files allowed")

    # access the cloud
    try:
        # init. B2 SDK
        info = InMemoryAccountInfo()
        b2_api = B2Api(info)
        # authentication
        b2_api.authorize_account("production", bb_key_id, bb_key)
        # get the bucket
        bucket = b2_api.get_bucket_by_name(bb_bucket_name)
    except Exception as ex:
        raise RuntimeError(f'failed to access the cloud storage: {ex}')

    # list of all file names from the raw-data bucket
    file_names = [file_info.file_name for file_info, _ in bucket.ls(recursive=True)]
    if file_name not in file_names:
        raise FileNotFoundError(f'selected dataset {file_name} NOT present in the bucket')

    # load the specific dataset
    try:
        # store data into in-memory object ('in_memory_data')
        in_memory_data = BytesIO()  # inmemory storage object
        bucket.download_file_by_name(file_name).save(
            in_memory_data)  # download data from specified bucket-file into 'in_memory_storage'
        in_memory_data.seek(0)  # rewind to the beginning
        # Load into pandas df
        if file_name.endswith('.csv'):
            dataset = pd.read_csv(in_memory_data)
        else:
            dataset = pd.read_excel(in_memory_data)
    except Exception as ex:
        raise RuntimeError(f'failed to download the dataset: {ex}')

    return dataset


# load selected dataset into a pd data-frame
def BB2_download(file_name,  # file name with extension (.csv or .xlsx)
                 output_folder,
                 bb_bucket_name=data_bucket_name,
                 bb_key_id=bucket_key_id,
                 bb_key=bucket_key):
    # check the file format
    extension = Path(file_name).suffix.lower()
    if extension not in [".csv", ".xlsx"]:
        raise ValueError(f"unsupported file type: '{extension}'. Only CSV and XLSX files allowed")

    # Check output folder and create it if necessary
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Combine folder and filename into the output path
    output_path = output_folder / Path(file_name).name

    # access the cloud
    try:
        # init. B2 SDK
        info = InMemoryAccountInfo()
        b2_api = B2Api(info)
        # authentication
        b2_api.authorize_account("production", bb_key_id, bb_key)
        # get the bucket
        bucket = b2_api.get_bucket_by_name(bb_bucket_name)
    except Exception as ex:
        raise RuntimeError(f'failed to access the cloud storage: {ex}')

    # list of all file names from the raw-data bucket
    file_names = [file_info.file_name for file_info, _ in bucket.ls(recursive=True)]
    if file_name not in file_names:
        raise FileNotFoundError(f'selected dataset {file_name} NOT present in the bucket')

    # download the specific dataset
    try:
        downloaded_file = bucket.download_file_by_name(file_name)
        downloaded_file.save_to(output_path)
    except Exception as ex:
        raise RuntimeError(f'failed to download the dataset: {ex}')