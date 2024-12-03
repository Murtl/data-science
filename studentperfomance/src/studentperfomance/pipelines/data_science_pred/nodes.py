import pandas as pd

def generate_predictions(predictor, test_data: pd.DataFrame, label_column: str):
    """
    Generate predictions from a trained AutoGluon model and combine with the test data
    and actual labels for evaluation.

    Args:
        predictor: The trained AutoGluon predictor.
        test_data (pd.DataFrame): The test dataset.
        label_column (str): The name of the target column.

    Returns:
        pd.DataFrame: A DataFrame containing test data, actual labels, and predictions.
    """
    # Ensure the target column exists in the test data
    if label_column not in test_data.columns:
        raise ValueError(f"The specified label column '{label_column}' is not in the test data.")

    # Separate the actual labels
    actual_labels = test_data[label_column]

    # Drop the target column from the test data to prevent data leakage
    test_data_features = test_data.drop(columns=[label_column])

    # Generate predictions
    predictions = predictor.predict(data=test_data_features)

    # Combine test data, actual labels, and predictions into a single DataFrame
    results_df = test_data.copy()
    results_df['Pred_Score'] = predictions
    results_df['Actual_Score'] = actual_labels

    return results_df
