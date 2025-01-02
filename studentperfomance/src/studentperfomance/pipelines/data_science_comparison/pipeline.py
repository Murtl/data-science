from kedro.pipeline import Pipeline, node
from .nodes import calculate_accuracies, visualize_accuracies, generate_best_model_visualizations, generate_feature_importance_plot

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=calculate_accuracies,
                inputs=["autogluon_tabular_predictions", "autogluon_tabular_nn_predictions", "autogluon_multimodal_predictions"],
                outputs="model_accuracies",
                name="calculate_accuracies_node",
            ),
            node(
                func=visualize_accuracies,
                inputs="model_accuracies",
                outputs="accuracy_visualization",
                name="visualize_accuracies_node",
            ),
            node(
                func=generate_best_model_visualizations,
                inputs=["model_accuracies", "student_performance_factors_test_data", "autogluon_tabular_model", "autogluon_tabular_nn_model", "autogluon_multimodal_model"],
                outputs=["best_model_leaderboard", "best_model_feature_importance"],
                name="generate_best_model_visualizations_node",
            ),
            node(
                func=generate_feature_importance_plot,
                inputs="best_model_feature_importance",
                outputs="best_model_feature_importance_plot",
                name="generate_feature_importance_plot_node",
            ),
        ]
    )
