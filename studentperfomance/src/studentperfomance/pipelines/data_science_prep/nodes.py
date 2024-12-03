import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(data: pd.DataFrame, train_size: float = 0.9, test_size: float = 0.1):
    """
    Splits the dataset into training and test sets without stratification.

    Args:
        data (pd.DataFrame): The dataset to split.
        train_size (float): Proportion of the data for training.
        test_size (float): Proportion of the data for testing.

    Returns:
        dict: A dictionary containing the split data as 'train' and 'test'.
    """
    # Ensure the proportions sum up to 1.0
    assert train_size + test_size == 1.0, "Train and test proportions must sum to 1.0."

    # Split into train and test
    train_data, test_data = train_test_split(
        data,
        test_size=test_size,
        random_state=42  # Set a random seed for reproducibility
    )

    return {
        "train": train_data,
        "test": test_data
    }