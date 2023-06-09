#!/usr/bin/env python3

from pathlib import Path
import traceback as tb

import click
import pandas as pd

from src.app_logger import logger
from src.eval import calc_metrics
from config import *

@click.command()
@click.option('--ground_truth', '-t', default='data.csv',
              type=click.Path(exists=True, path_type=Path), help = 'CSV file with ground truth.')
@click.option('--predictions', '-p', default='predict.csv',
              type=click.Path(exists=True, path_type=Path), help = 'CSV file with predictions.')
@click.option('--save_errors', '-e', is_flag = True, default = False,
              help = f'Save errors to {ERRORS_FILE} file.')
def evaluation(ground_truth: Path, predictions: Path, save_errors: bool) -> None:
    """
    Log classification metrics:
        - accuracy
        - confusion matrix
    Save errors list.
    CSV files format:
        'path': path to image
        'class_number': class number
    """
    try:
        ground_truth_df = pd.read_csv(ground_truth)
        assert list(ground_truth_df.columns) == ['path', 'class_number'], "Ground truth file with incorrect columns. Correct columns names: ['path', 'class_number']"
        predictions_df = pd.read_csv(predictions)
        assert list(predictions_df.columns) == ['path', 'class_number'], "Predictions file with incorrect columns. Correct columns names: ['path', 'class_number']"
        accuracy, conf_matrix, errors_df = calc_metrics(ground_truth_df, predictions_df)
        logger.info(f"Accuracy: {accuracy}")
        logger.info(f"Confusion matrix: \n{conf_matrix}")
        if save_errors:
            errors_df.to_csv(ERRORS_FILE, index=False)
            logger.info(f"Save errors to {ERRORS_FILE}.")
    except AssertionError as message:
        logger.info(f"evaluation: some errors: {message}.")
    except:
        logger.info(f"evaluation: some errors occurred! Reason: {tb.format_exc()}")

if __name__ == '__main__':
    evaluation()

