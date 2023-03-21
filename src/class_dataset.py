
from dataclasses import dataclass
import traceback as tb

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image

from config import *
from src.app_logger import logger


@dataclass
class TrainDataset(Dataset):
    """
    Get image samples and their labeles from specified list
    """
    img_files_df: pd.DataFrame  # DataFrame with data info: 'path', 'class_number'
    get_random: bool=True  # Shuffle data or not

    def __post_init__(self):
        try:
            assert list(self.img_files_df.columns) == ['path', 'class_number'], "Dataset file with incorrect columns. Correct columns name: ['path', 'class_number']"
            if self.get_random:
                self.img_files_df = self.img_files_df.sample(frac=1, random_state=RANDOM_SEED)
                self.img_files_df.reset_index(drop=True, inplace=True)
            self.class_list = self.img_files_df['class_number'].unique()
            self.transform = transforms.ToTensor()
        except AssertionError as message:
            logger.info(f"TrainDataset: some errors: {message}.")
            exit(0)
        except:
            logger.info(f"TrainDataset: some errors occurred. Reason: {tb.format_exc()}")
            exit(0)

    def __len__(self):
        return len(self.img_files_df)

    def __getitem__(self, idx):
        """
        :return: sample: image, label
        """
        fail = True
        counter = 0
        while fail:
            try:
                image = Image.open(self.img_files_df.loc[idx, 'path'])
                image = self.transform(image)
                label = self.img_files_df.loc[idx, 'class_number']
                fail = False
            except:
                logger.info(f"TrainDataset: error reading image file: {self.img_files_df.loc[idx, 'path']}")
                idx += 1
                counter += 1
                if counter >= MAX_COUNT_OF_ERROR_IMAGES:
                    exit(0)
        return image, label


@dataclass
class InferenceDataset(Dataset):
    """
    Get image samples from specified list
    """
    img_files_df: pd.DataFrame  # DataFrame with data info: 'path'

    def __post_init__(self):
        try:
            assert 'path' in list(self.img_files_df.columns), "Dataset file with incorrect columns. No column 'path'."
            self.transform = transforms.ToTensor()
        except AssertionError as message:
            logger.info(f"InferenceDataset: some errors: {message}.")
            exit(0)
        except:
            logger.info(f"InferenceDataset: some errors occurred. Reason: {tb.format_exc()}")
            exit(0)

    def __len__(self):
        return len(self.img_files_df)

    def __getitem__(self, idx):
        """
        :return: sample: image, path_to_image
        """
        fail = True
        counter = 0
        while fail:
            try:
                image = Image.open(self.img_files_df.loc[idx, 'path'])
                image = self.transform(image)
                fail = False
            except:
                logger.info(f"InferenceDataset: Error reading image file: {self.img_files_df.loc[idx, 'path']}")
                idx += 1
                counter += 1
                if counter >= MAX_COUNT_OF_ERROR_IMAGES:
                    exit(0)
        return image, self.img_files_df.loc[idx, 'path']

