
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from src.app_logger import logger
from src.class_dataset import TrainDataset
from src.class_model import CustomModel
from config import *


def train(dataset: Path, model_path:Path) -> None:
    """
    Train model with specified data. Save model to specified path.
    :param dataset: dataset csv-file:
        path_to_the_image, class_number
    :param model_path: path for model saving
    :return:
    """
    logger.info("Start data reading.")
    dataset_df = pd.read_csv(dataset)
    train_df = dataset_df.sample(frac=TRAIN_TEST_SPLIT)
    test_df = dataset_df.drop(train_df.index)
    train_data = TrainDataset(train_df)
    test_data = TrainDataset(test_df)
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE)
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE)


def inference(input_data: Path, model_path: Path, predict_file: Path) -> None:
    """
    Make inference for specified data and model. Save results to the csv-file.
    :param input_data: CSV file with input data
    :param model_path: Path to the classification model
    :param predict_file: CSV file for saving predictions
    :return:
    """
    logger.info("Start data reading.")
    dataset_df = pd.read_csv(dataset)
    train_df = dataset_df.sample(frac=TRAIN_TEST_SPLIT)
    test_df = dataset_df.drop(train_df.index)
    train_data = TrainDataset(train_df)
    test_data = TrainDataset(test_df)
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE)
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE)
