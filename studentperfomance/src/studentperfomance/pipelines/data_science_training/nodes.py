from autogluon.tabular import TabularPredictor
from autogluon.multimodal import MultiModalPredictor
import pandas as pd

def train_tabular_model(train_data: pd.DataFrame, label_column: str):
    """
    Train a tabular model using AutoGluon.
    """
    predictor = TabularPredictor(label=label_column).fit(
        train_data, 
        presets='best_quality',
        time_limit= 4 * 3600,  # Set time limit to 4 hours (4 * 3600 seconds)
        ag_args_fit={'num_gpus': 1}  # Enable GPU usage if available
    )
    return predictor
    

def train_tabular_nn_model(train_data: pd.DataFrame, label_column: str):
    """
    Train a tabular model focused on neural networks using AutoGluon.
    """
    predictor = TabularPredictor(label=label_column).fit(
        train_data,
        presets="best_quality",
        hyperparameters={"NN_TORCH": {}},
        time_limit= 4 * 3600,  # Set time limit to 4 hours (4 * 3600 seconds)
        ag_args_fit={'num_gpus': 1}  # Enable GPU usage if available
    )
    return predictor

def train_multimodal_model(train_data: pd.DataFrame, label_column: str):
    """
    Train a multimodal model using AutoGluon.
    """
    predictor = MultiModalPredictor(label=label_column).fit(
        train_data,
        presets='best_quality',
        time_limit= 4 * 3600,
        hyperparameters={
            'env.num_gpus': 1,            # GPU verwenden
        }
    )
    return predictor
