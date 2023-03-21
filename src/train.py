
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

    logger.info("Start model loading.")
    model = CustomModel(target_size=CLASSES_NUMBER)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()

    logger.info(f"Start training with {device}.")
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, factor=SCHEDULER_LR_MULT_FACTOR, patience=SCHEDULER_PLATO_SIZE)
    criterion = CrossEntropyLoss()
    best_tp = 0
    for epoch in range(EPOCHS):
        curr_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Start {epoch} epoch. LR: {curr_lr}")
        train_loss = 0
        train_tp = 0
        for batch in tqdm(train_loader, desc=f"Train, epoch {epoch}"):
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            y_pred = torch.argmax(y_pred, axis=1)
            train_tp += (y_pred == y_batch).type(torch.float).sum().item()
        logger.info(f"Train loss: {train_loss / len(train_loader):.5f}. Train accuracy: {100 * train_tp / len(train_data):.2f}%")

        valid_loss = 0
        valid_tp = 0
        for batch in tqdm(test_loader, desc=f"Validation, epoch {epoch}"):
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)
            with torch.no_grad():
                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch)
                valid_loss += loss.item()
                y_pred = torch.argmax(y_pred, axis=1)
                valid_tp += (y_pred == y_batch).type(torch.float).sum().item()
        logger.info(f"Validation loss: {valid_loss / len(test_loader):.5f}. Validation accuracy: {100 * valid_tp / len(test_data):.2f}%")
        if valid_tp > best_tp:
            best_tp = valid_tp
            logger.info(f"Save model to: {model_path}")
            torch.save(model.state_dict(), model_path)
        scheduler.step(valid_loss / len(test_loader))

    logger.info(f"Training is finished. Best accuracy: {best_tp / len(test_loader):.3f}. Best model saved to {model_path}.")

