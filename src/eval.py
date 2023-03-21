
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

def calc_metrics(ground_truth_df: pd.DataFrame, predict_df: pd.DataFrame):
    join_df = pd.merge(ground_truth_df, predict_df, on='path')
    accuracy = accuracy_score(join_df['class_number_x'].values, join_df['class_number_y'].values)
    conf_matrix = confusion_matrix(join_df['class_number_x'].values, join_df['class_number_y'].values)
    return accuracy, conf_matrix
