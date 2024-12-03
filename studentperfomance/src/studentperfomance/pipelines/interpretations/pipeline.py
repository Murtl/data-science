from kedro.pipeline import Pipeline, node
from .nodes import generate_best_model_visualizations

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=generate_best_model_visualizations,
            inputs=["model_accuracies", "student_performance_factors_test_data", "autogluon_tabular_model", "autogluon_tabular_nn_model", "autogluon_multimodal_model"],
            outputs="best_model_visualisation",
            name="generate_best_model_visualizations_node",
        ),
    ])
