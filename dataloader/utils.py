
# dataloader/utils.py

import time
import requests
import gzip
import shutil
import os
import gdown
import struct
import numpy as np
import tarfile
from pathlib import Path
import pickle
import csv
import zipfile

def timer(func):
    """
    A decorator that measures the execution time of a function and prints the elapsed time.

    Args:
        func (function): The function to be timed.

    Returns:
        function: The wrapped function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        print(f"Function '{func.__name__}' took {time.time() - start_time:.2f}s to complete.")
        return result
    return wrapper

def download_file(url, dest_path):
    """
    Downloads a file from a given URL or copies a local file to the destination path.

    If the URL is a local file path, it copies the file to the destination path.
    If the URL is a remote file, it downloads the file using `gdown`.
    The function also handles decompression of `.gz`, `.tar`, and `.zip` files.

    Args:
        url (str): The URL or local file path of the file to be downloaded.
        dest_path (str): The destination path where the file will be saved.

    Returns:
        None
    """
    if os.path.exists(url):
        # If the URL is a local file path, copy it to the destination path
        shutil.copy(url, dest_path)
        print(f'Copied local file to {dest_path}')
    else:
        # Otherwise, download the file from the URL    
        gdown.download(url, dest_path, quiet=False)
        print(f'Destination path = {dest_path}')
        if dest_path.endswith('.gz'):
            with gzip.open(dest_path, 'rb') as f_in:
                with open(dest_path[:-3], 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f'Data extracted successfully for file {dest_path[:-3]}')
        elif dest_path.endswith('.tar'):
            if tarfile.is_tarfile(dest_path):
                with tarfile.open(dest_path) as tar:
                    to_extract = Path(dest_path).parents[0]
                    tar.extractall(path=to_extract)
                    print(f"Extracted to {to_extract}")
            else:
                print("The file is not a valid tar file.")
        elif dest_path.endswith('.zip'):
            with zipfile.ZipFile(dest_path, 'r') as zip_ref:
                zip_ref.extractall(Path(dest_path).parents[0])
            print(f'Data extracted successfully for file {dest_path}')

def read_idx(filename):
    """
    Reads an IDX file and returns the data as a NumPy array.

    IDX files are used to store multidimensional arrays of numeric data.
    This function supports reading both images (magic number 2051) and labels (magic number 2049).

    Args:
        filename (str): The path to the IDX file.

    Returns:
        numpy.ndarray: The data read from the IDX file.

    Raises:
        ValueError: If the IDX file has an unexpected magic number.
    """
    with open(filename, 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        if magic == 2051:
            rows, cols = struct.unpack(">II", f.read(8))
            data = np.fromfile(f, dtype=np.uint8).reshape(size, rows, cols)
        elif magic == 2049:
            data = np.fromfile(f, dtype=np.uint8)
        else:
            raise ValueError("Invalid IDX file: unexpected magic number.")
    return data

def unpickle(file):
    """
    Unpickles a file and returns the deserialized object.

    Args:
        file (str): The path to the pickled file.

    Returns:
        dict: The deserialized object.
    """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def read_csv(filename):
    """
    Reads a CSV file and returns its contents as a list of rows.

    Args:
        filename (str): The path to the CSV file.

    Returns:
        list: A list of rows, where each row is a list of strings.
    """
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        return list(reader)

def read_text(filename):
    """
    Reads a text file and returns its contents as a list of lines.

    Args:
        filename (str): The path to the text file.

    Returns:
        list: A list of strings, where each string is a line from the text file.
    """    
    with open(filename, 'r') as f:
        return f.readlines()