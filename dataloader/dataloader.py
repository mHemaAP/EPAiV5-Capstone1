# dataloader/dataloader.py

import os
import sys
import requests
from collections import namedtuple
from contextlib import contextmanager
from .preprocessors import default_preprocess, augment_image, augment_text
from .utils import download_file, timer, read_idx, unpickle, read_csv, read_text
import numpy as np
import shutil
import matplotlib.pyplot as plt

DataSample = namedtuple('DataSample', ['features', 'label'])
"""
A named tuple representing a data sample with features and an optional label.

Attributes:
    features (Any): The features of the data sample.
    label (Any): The label associated with the data sample (optional).
"""

class DataLoader:
    """
    A class for loading and preprocessing datasets.

    Attributes:
        dataset_name (str): The name of the dataset to load.
        batch_size (int): The size of each batch of data.
        shuffle (bool): Whether to shuffle the data before iterating.
        kwargs (dict): Additional keyword arguments.
        data (list): The preprocessed data samples.
        index (int): The current index for iterating through the data.
    """
    def __init__(self, dataset_name='MNIST', batch_size=32, shuffle=True, **kwargs):
        """
        Initializes the DataLoader with the specified dataset name, batch size,
        and other parameters.

        Args:
            dataset_name (str): The name of the dataset to load.
            batch_size (int): The size of each batch of data.
            shuffle (bool): Whether to shuffle the data before iterating.
            **kwargs: Additional keyword arguments.
        """
        self.dataset_name = sys.intern(dataset_name)  # Intern dataset name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.kwargs = kwargs
        self.data = []
        self.index = 0
        self.load_data()
    
    @timer
    def load_data(self):
        """
        Loads the dataset and preprocesses it.

        If the dataset does not exist locally, it is downloaded. The raw data 
        is then read and preprocessed.
        """
        if not os.path.exists(f'datasets/{self.dataset_name}'):
            self.download_dataset()
        raw_data = self.read_data()
        self.data = self.preprocess_data(raw_data)
    
    @timer
    def download_dataset(self):
        """
        Downloads the dataset if it is not already present locally.

        The dataset is downloaded based on the dataset name. Different datasets
        have different URLs and file formats.
        """
        print(f"Downloading {self.dataset_name} dataset...")
        if self.dataset_name == 'MNIST':
            urls = {
                'train_images': 'https://drive.google.com/uc?export=download&id=1ruFYL2hHgetFc6hLFE87aSS9GQQAgav9',
                'train_labels': 'https://drive.google.com/uc?export=download&id=1ILIdcDlpcs55lkQ1ycot58S4TQPu2_sx',
                'test_images': 'https://drive.google.com/uc?export=download&id=1AOW0gGEgQHU4EXrAG5o9m-UquA4R5aHP',
                'test_labels': 'https://drive.google.com/uc?export=download&id=12nE2NfMEz0SOVA0Pb_aIsGPvy6z9brw4'
            }
            for folder_name, url in urls.items():
                folder_name = sys.intern(folder_name)  # Intern folder name
                os.makedirs(f'datasets/{self.dataset_name}/{folder_name}', exist_ok=True)
                dest_path = f'datasets/{self.dataset_name}/{folder_name}/file.gz'
                download_file(url, dest_path)
        elif self.dataset_name == 'CIFAR-10':
            os.makedirs(f'datasets/{self.dataset_name}', exist_ok=True)
            urls = {
                'data' : 'https://drive.google.com/uc?export=download&id=1Nh71Y_31pP2qu4KuNMqBavBg7h-WDMNm'
            }
            dest_path = f'datasets/{self.dataset_name}/file.tar'
            download_file(urls['data'], dest_path)
        elif self.dataset_name == 'smsspamcollection':
            os.makedirs(f'datasets/{self.dataset_name}', exist_ok=True)
            urls = {
                'data' : 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
            }
            dest_path = f'datasets/{self.dataset_name}/smsspamcollection.zip'
            download_file(urls['data'], dest_path)

        elif self.dataset_name == 'nyc_csv_data':
            os.makedirs(f'datasets/{self.dataset_name}', exist_ok=True)
            local_file_path = 'nyc_parking_tickets_extract-1.csv'  # Path to the local file in the root folder
            dest_path = f'datasets/{self.dataset_name}/'
            download_file(local_file_path, dest_path)
    
    def read_data(self):
        """
        Reads the raw data from the dataset files.

        The method reads data based on the dataset name. Different datasets have different file formats and reading methods.

        Returns:
            list: A list of DataSample objects containing the raw data.
        """
        if self.dataset_name == 'MNIST': 
            dataset_to_read = {
                'train_images' : f'datasets/{self.dataset_name}/train_images/file',
                'train_labels' : f'datasets/{self.dataset_name}/train_labels/file',
                'test_images' : f'datasets/{self.dataset_name}/test_images/file',
                'test_labels' : f'datasets/{self.dataset_name}/test_labels/file'
            }
            train_data, train_labels = None, None
            for type, path in dataset_to_read.items():
                type = sys.intern(type)  # Intern type
                if os.path.exists(path):
                    if type == 'train_images':
                        train_data = read_idx(path)
                    elif type == 'train_labels':
                        train_labels = read_idx(path)
                else:
                    raise ValueError(f"In {self.dataset_name} {type} doesn't exists")
            return [DataSample(features=train_data[i], label=train_labels[i]) for i in range(len(train_data))]
        elif self.dataset_name == 'CIFAR-10':
            cifar_path = f'datasets/{self.dataset_name}/cifar-10-batches-py'
            all_files = os.listdir(cifar_path)
            train_img_files = [f for f in all_files if f.startswith('data_batch_')]
            train_data = []
            for img_batch in train_img_files:
                img_batch = sys.intern(img_batch)  # Intern img_batch
                data_dict = unpickle(os.path.join(cifar_path,img_batch))
                img_data = data_dict[b'data']
                img_label = data_dict[b'labels']
                img_data = img_data.reshape(-1, 3, 32, 32)
                img_data = np.transpose(img_data, (0, 2, 3, 1))
                train_data.append(DataSample(features=img_data.tolist(), label=img_label))
            print(f'Length of CIFAR-10 data = {len(train_data)}')
            return train_data
        elif self.dataset_name == 'smsspamcollection':
            text_path = f'datasets/{self.dataset_name}/SMSSpamCollection'
            if os.path.exists(text_path):
                text_data = read_text(text_path)
                return [DataSample(features=text, label=None) for text in text_data]
            else:
                # Handle the case where the text data doesn't exist
                print(f"Warning: In {self.dataset_name} text data doesn't exist. Returning empty list.")
                return []
        elif self.dataset_name == 'nyc_csv_data':
            csv_path = f'datasets/{self.dataset_name}/nyc_parking_tickets_extract-1.csv'
            if os.path.exists(csv_path):
                csv_data = read_csv(csv_path)
                return [DataSample(features=row, label=None) for row in csv_data]
            else:
                # Handle the case where the CSV data doesn't exist
                print(f"Warning: In {self.dataset_name} CSV data doesn't exist. Returning empty list.")
                return []
    
    def preprocess_data(self, data):
        """
        Preprocesses the raw data using the specified preprocessing function.

        Args:
            data (list): A list of DataSample objects containing the raw data.

        Returns:
            list: A list of preprocessed data samples.
        """
        preprocess_func = self.kwargs.get('preprocess_func', default_preprocess)
        if self.dataset_name in ['MNIST', 'CIFAR-10']:
            return [preprocess_func(sample.features) for sample in data]
        elif self.dataset_name in ['tiny_web_data', 'nyc_parking_tickets_extract-1']:
            return [preprocess_func(sample.features) for sample in data]
    
    def __iter__(self):
        """
        Initializes the iterator for the DataLoader.

        If shuffle is enabled, the data is shuffled before iterating.

        Returns:
            DataLoader: The DataLoader instance.
        """
        self.index = 0
        if self.shuffle:
            import random
            random.shuffle(self.data)
        return self
    
    def __next__(self):
        """
        Returns the next batch of data.

        Returns:
            list: A list of DataSample objects representing the next batch of data.

        Raises:
            StopIteration: If there are no more batches to return.
        """
        if self.index < len(self.data):
            batch = self.data[self.index:self.index + self.batch_size]
            self.index += self.batch_size
            return batch
        else:
            raise StopIteration

