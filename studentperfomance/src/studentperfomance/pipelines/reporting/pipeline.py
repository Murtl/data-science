from kedro.pipeline import Pipeline, node
from .nodes import calculate_accuracies, visualize_accuracies

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
        ]
    )
