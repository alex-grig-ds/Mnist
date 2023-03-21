
from pathlib import Path

from tqdm import tqdm
import pandas as pd

def create_train_csv_from_data_folder(path_to_folder: str, csv_file: str, dataset_size: int=0) -> bool:
    """
    Read data folder with class subfolders and create csv file with format:
        path_to_image, class_name
    Class_names equal to subfolders names.
    :param path_to_folder:
    :param csv_file:
    :param dataset_size: qnty of elements of different classes in dataset. If == 0 - include all elements.
    :return: success of operation
    """
    data_folder = Path(path_to_folder)
    if not data_folder.is_dir():
        return False

    success = True
    dataset_df = pd.DataFrame(columns = ['path', 'class_number'])
    for subfolder in tqdm(sorted(data_folder.iterdir())):
        if subfolder.is_dir():
            class_name = str(subfolder.name)
            for data_file in subfolder.glob('*.*'):
                dataset_df.loc[len(dataset_df), :] = [str(data_file), class_name]
    if len(dataset_df) == 0:
        success = False
    else:
        csv_path = Path(csv_file)
        csv_path.parent.mkdir(exist_ok=True, parents=True)
        if dataset_size > 0:
            dataset_df = dataset_df.sample(dataset_size)
        dataset_df.to_csv(str(csv_path), index=False)
    return success


def create_inference_csv_from_data_folder(path_to_folder: str, csv_file: str, dataset_size: int=0) -> bool:
    """
    Read data folder with class subfolders and create csv file with format:
        path_to_image, class_name
    Class_names equal to subfolders names.
    :param path_to_folder:
    :param csv_file:
    :param dataset_size: qnty of elements of different classes in dataset. If == 0 - include all elements.
    :return: success of operation
    """
    data_folder = Path(path_to_folder)
    if not data_folder.is_dir():
        return False

    success = True
    dataset_df = pd.DataFrame(columns = ['path'])
    for subfolder in tqdm(sorted(data_folder.iterdir())):
        if subfolder.is_dir():
            for data_file in subfolder.glob('*.*'):
                dataset_df.loc[len(dataset_df), 'path'] = str(data_file)
    if len(dataset_df) == 0:
        success = False
    else:
        csv_path = Path(csv_file)
        csv_path.parent.mkdir(exist_ok=True, parents=True)
        if dataset_size > 0:
            dataset_df = dataset_df.sample(dataset_size)
        dataset_df.to_csv(str(csv_path), index=False)
    return success
