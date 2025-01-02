import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_accuracies(model_1_predictions: pd.DataFrame, 
                         model_2_predictions: pd.DataFrame, 
                         model_3_predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the accuarcy of each model based on the predictions on the test_data.
    """
    accuracies = {
        "Model": [],
        "Accuracy": []
    }
    
    for i, (predictions, model_name) in enumerate(zip(
        [model_1_predictions, model_2_predictions, model_3_predictions],
        ["Tabular Predictor", "Tabular Predictor NN", "Mutli Modal"]
    )):
        acc = accuracy_score(predictions["Actual_Score"], predictions["Pred_Score"])
        accuracies["Model"].append(model_name)
        accuracies["Accuracy"].append(acc)
    
    return pd.DataFrame(accuracies)

def visualize_accuracies(model_accuracies: pd.DataFrame) -> str:
    """
    Visualizes the Accuracy of each model and saves a picture
    """
    plt.figure(figsize=(10, 6))
    plt.bar(model_accuracies["Model"], model_accuracies["Accuracy"])
    plt.title("Model Accuracies")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    
    return plt


def generate_best_model_visualizations(model_accuracies: pd.DataFrame, student_performance_factors_test_data: pd.DataFrame, autogluon_tabular_model, autogluon_tabular_nn_model, autogluon_multimodal_model) -> str:
    """
    Generates visualizations for the best model.

    Args:
        model_accuracies: DataFrame containing model names and their accuracies.
        models: A dictionary mapping model names to their respective objects.
        student_performance_factors_test_data: Test dataset for feature importance calculation.

    Returns:
        File path of the plot containing all useful information about the best model.
    """
    # Since in this case the TabularPredictor is the best one, only the visualization for this is implemented the rest is tbd
    # best_model_name = select_best_model(model_accuracies)
    # if best_model_name in ["Tabular Predictor"]:
    #     return generate_visualisation_for_tabular_predictor_model(autogluon_tabular_model, student_performance_factors_test_data, best_model_name)
    # if best_model_name in ["Tabular Predictor NN"]:   
    #     return generate_visualisation_for_tabular_predictor_model(autogluon_tabular_nn_model, student_performance_factors_test_data, best_model_name)
    # else:
    #     return generate_visualisation_for_multimodal_predictor_model(autogluon_multimodal_model, best_model_name)


    best_model_leaderboard = generate_leaderboard_table(autogluon_tabular_model)
    feature_importance_table = generate_feature_importance_table(autogluon_tabular_model, student_performance_factors_test_data)

    return best_model_leaderboard, feature_importance_table

def select_best_model(model_accuracies: pd.DataFrame) -> str:
    """
    Selects the best model based on accuracy.

    Args:
        model_accuracies: DataFrame containing model names and their accuracies.

    Returns:
        The name of the best model.
    """
    best_model = model_accuracies.loc[model_accuracies['Accuracy'].idxmax(), 'Model']
    return best_model

def generate_leaderboard_table(model) -> pd.DataFrame:
    """
    Retrieves the leaderboard.

    Args:
        model: The Tabular Predictor model object.

    Returns:
        DataFrame of the leaderboard entries.
    """
    leaderboard_df = model.leaderboard()
    return leaderboard_df

def generate_feature_importance_table(model, student_performance_factors_test_data: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a table of the sorted feature importance.

    Args:
        model: The Tabular Predictor model object.
        student_performance_factors_test_data: Test dataset for feature importance calculation.

    Returns:
        Table containing the feature importance.
    """
    feature_importance_df = model.feature_importance(student_performance_factors_test_data)
    feature_importance_df = feature_importance_df.reset_index()
    feature_importance_df.rename(columns={"index": "feature"}, inplace=True)

    importance_sorted = feature_importance_df.sort_values(by="importance", ascending=False)

    return importance_sorted

def generate_feature_importance_plot(feature_importance_df: pd.DataFrame):
    """
    Creates a plot of the sorted feature importance.

    Args:
        model: The Tabular Predictor model object.
        student_performance_factors_test_data: Test dataset for feature importance calculation.

    Returns:
        Matplotlib figure containing the sorted feature importance plot.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.barplot(
        x=feature_importance_df["importance"],
        y=feature_importance_df["feature"],
        ax=ax,
        palette="viridis"
    )
    ax.set_title("Feature Importance", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)

    plt.tight_layout()

    return fig

# tbd if the multimodal gets the best results (for now the tabular_predictor visualization is enough)
# def generate_visualisation_for_multimodal_predictor_model(model, best_model_name: str) -> str:
#     """
#     Creates a plot with the output of model.fit_summary() and includes the model name.

#     Args:
#         model: The MultiModal Predictor model object.
#         model_name: The name of the model.

#     Returns:
#         File path of the saved plot.
#     """
#     return ??

