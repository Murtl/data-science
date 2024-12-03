import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    best_model_name = select_best_model(model_accuracies)

    if best_model_name in ["Tabular Predictor"]:
        return generate_visualisation_for_tabular_predictor_model(autogluon_tabular_model, student_performance_factors_test_data, best_model_name)
    if best_model_name in ["Tabular Predictor NN"]:   
        return generate_visualisation_for_tabular_predictor_model(autogluon_tabular_nn_model, student_performance_factors_test_data, best_model_name)
    else:
        return generate_visualisation_for_multimodal_predictor_model(autogluon_multimodal_model, best_model_name)


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


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def generate_visualisation_for_tabular_predictor_model(
    model, student_performance_factors_test_data: pd.DataFrame, best_model_name: str
) -> str:
    """
    Creates a visualization for the Tabular Predictor model including:
    1. The model name.
    2. The top 15 entries of the leaderboard (restricted to the first 5 columns).
    3. The full feature importance table.
    4. A plot of sorted feature importance.

    Args:
        model: The Tabular Predictor model object.
        student_performance_factors_test_data: Test dataset for feature importance calculation.
        best_model_name: Name of the model.

    Returns:
        File path of the saved visualization.
    """
    # 1. Model Name
    fit_summary_title = f"Model: {best_model_name}"

    # 2. Leaderboard Entries
    leaderboard_df = model.leaderboard()
    leaderboard_top_15 = leaderboard_df.iloc[:15, :5]  # Top 15 rows, first 5 columns only

    # 3. Feature Importance Table
    feature_importance_df = model.feature_importance(student_performance_factors_test_data)
    feature_importance_df = feature_importance_df.reset_index()  # Ensure features are included as a column
    feature_importance_df.rename(columns={"index": "feature"}, inplace=True)

    # 4. Sort Feature Importance for Visualization
    importance_sorted = feature_importance_df.sort_values(by="importance", ascending=False)

    # Create a figure with enough space for all components
    fig, axes = plt.subplots(nrows=3, figsize=(14, 25), gridspec_kw={"height_ratios": [1, 2, 3]})
    fig.subplots_adjust(hspace=0.5)  # Add spacing between sections

    # Add the model name
    axes[0].axis("off")
    axes[0].text(0.02, 0.9, fit_summary_title, fontsize=16, weight="bold", va="top", ha="left")

    # Add the leaderboard (Top 15, first 5 columns)
    leaderboard_text = f"Top 15 Leaderboard Entries (First 5 Columns):\n\n{leaderboard_top_15.to_string(index=False)}"
    axes[0].text(0.02, 0.6, leaderboard_text, fontsize=12, va="top", ha="left", family="monospace")

    # Add the full feature importance table
    feature_importance_text = f"Feature Importance Table:\n\n{feature_importance_df.to_string(index=False)}"
    axes[1].axis("off")
    axes[1].text(0.02, 1, feature_importance_text, fontsize=12, va="top", ha="left", family="monospace")

    # Plot Sorted Feature Importance
    sns.barplot(
        x=importance_sorted["importance"], 
        y=importance_sorted["feature"], 
        ax=axes[2], 
        palette="viridis"
    )
    axes[2].set_title("Feature Importance", fontsize=14, fontweight="bold")
    axes[2].set_xlabel("Importance", fontsize=12)
    axes[2].set_ylabel("Feature", fontsize=12)

    plt.tight_layout()

    return plt




def generate_visualisation_for_multimodal_predictor_model(model, best_model_name: str) -> str:
    """
    Creates a plot with the output of model.fit_summary() and includes the model name.

    Args:
        model: The MultiModal Predictor model object.
        model_name: The name of the model.

    Returns:
        File path of the saved plot.
    """
    # Get fit summary text
    fit_summary_text = model.fit_summary()

    # Combine model name with fit summary
    visualization_text = f"Model: {best_model_name}\n\n{fit_summary_text}"

    # Create a figure to display the fit summary
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis("off")  # No axes for text
    ax.text(0, 1, visualization_text, fontsize=10, ha="left", va="top", wrap=True)

    plt.tight_layout()

    return plt

