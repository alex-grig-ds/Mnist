#!/usr/bin/env/python3

from pathlib import Path
import traceback as tb

import click

from src.app_logger import logger
from src.train import train
from src.inference import inference

@click.command()
@click.option('--mode', '-m', default='inference',
              type=click.Choice(['train', 'inference']), help = 'Working mode.')
@click.option('--dataset', '-d', default='data.csv',
              type=click.Path(path_type=Path), help = 'CSV file with training data description.')
@click.option('--model', '-md', required=True,
              type=click.Path(path_type=Path), help = 'Path to the classification model.')
@click.option('--input', 'input_', '-i', default='input.csv',
              type=click.Path(path_type=Path), help = 'CSV file with input data.')
@click.option('--output', '-o',
              type=click.Path(path_type=Path), help = 'CSV file for saving predictions.')
def mnist(mode: str, dataset: Path, model: Path, input_: Path, output: Path):
    """
    Classification with MNIST dataset.
    If mode = 'train' - train with specified dataset.
    If mode = 'inference' - make inference for specified data
    CSV files format: path_to_image, class_number
    """
    try:
        if mode == 'train':
            assert dataset.is_file(), 'Dataset file is missing'
            model.parent.mkdir(exist_ok=True, parents=True)
            logger.info("MNIST: start training.")
            train(dataset, model)
        elif mode == 'inference':
            assert input_.is_file(), 'Input file is missing'
            assert model.is_file(), 'Model file is missing'
            output.parent.mkdir(exist_ok=True, parents=True)
            logger.info("MNIST: start inference.")
            inference(input_, model, output)
    except AssertionError as message:
        logger.info(f"mnist: some errors: {message}.")
    except:
        logger.info(f"mnist: some errors occurred! Reason: {tb.format_exc()}")

if __name__ == '__main__':
    mnist()

