from kedro.pipeline import Pipeline, node, pipeline
from .nodes import generate_predictions

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=generate_predictions,
                inputs=dict(
                    predictor="autogluon_tabular_model",
                    test_data="student_performance_factors_test_data",
                    label_column="params:label_column"
                ),
                outputs="autogluon_tabular_predictions",
                name="generate_tabular_predictions_node",
            ),
            node(
                func=generate_predictions,
                inputs=dict(
                    predictor="autogluon_tabular_nn_model",
                    test_data="student_performance_factors_test_data",
                    label_column="params:label_column"
                ),
                outputs="autogluon_tabular_nn_predictions",
                name="generate_tabular_nn_predictions_node",
            ),
            node(
                func=generate_predictions,
                inputs=dict(
                    predictor="autogluon_multimodal_model",
                    test_data="student_performance_factors_test_data",
                    label_column="params:label_column"
                ),
                outputs="autogluon_multimodal_predictions",
                name="generate_multimodal_predictions_node",
            ),
        ]
    )
