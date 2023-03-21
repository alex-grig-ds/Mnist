
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from src.app_logger import logger
from src.class_dataset import InferenceDataset
from src.class_model import CustomModel
from config import *


def inference(input_data: Path, model_path: Path, predict_file: Path) -> None:
    """
    Make inference for specified data and model. Save results to the csv-file.
    :param input_data: CSV file with input data
    :param model_path: Path to the classification model
    :param predict_file: CSV file for saving predictions
    :return:
    """
    logger.info("Start data reading.")
    dataset_df = pd.read_csv(input_data)
    inference_data = InferenceDataset(dataset_df)
    data_loader = DataLoader(dataset=inference_data, batch_size=INFERENCE_BATCH_SIZE)

    logger.info("Start model loading.")
    model = CustomModel(target_size=CLASSES_NUMBER)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    logger.info("Start prediction.")
    output_df = pd.DataFrame(columns = ['path', 'class_number'])
    for batch in tqdm(data_loader):
        images = batch[0].to(device)
        predict = model(images)
        predict = torch.argmax(predict, axis=1).to('cpu')
        for idx, img_file in enumerate(batch[1]):
            output_df.loc[len(output_df), :] = [img_file, predict[idx].item()]
    output_df.to_csv(predict_file, index=False)
    logger.info(f"Save predictions to {str(predict_file)}")

