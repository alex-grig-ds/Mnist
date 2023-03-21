
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

def calc_metrics(ground_truth_df: pd.DataFrame, predict_df: pd.DataFrame) -> (float, np.array, pd.DataFrame):
    """
    Calc metrics:
        - accuracy
        - confusion matrix
    :param ground_truth_df: csv-file with base data
    :param predict_df: csv-file with predictions
    :return:
        - accuracy
        - confusion matrix
        - DataFrame with errors:
            'path': path to file
            'label': ground truth class number
            'predict': predicted class number
    """
    join_df = pd.merge(ground_truth_df, predict_df, on='path')
    accuracy = accuracy_score(join_df['class_number_x'].values, join_df['class_number_y'].values)
    conf_matrix = confusion_matrix(join_df['class_number_x'].values, join_df['class_number_y'].values)
    errors_df = join_df.loc[join_df['class_number_x'] != join_df['class_number_y'], :]
    errors_df = errors_df.rename(columns={'class_number_x': 'label', 'class_number_y': 'predict'})
    return accuracy, conf_matrix, errors_df
