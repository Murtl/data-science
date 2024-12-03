from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_tabular_model, train_tabular_nn_model, train_multimodal_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_tabular_model,
                inputs=dict(
                    train_data="student_performance_factors_train_data",
                    label_column="params:label_column"
                ),
                outputs="autogluon_tabular_model",
                name="train_tabular_model_node",
            ),
            node(
                func=train_tabular_nn_model,
                inputs=dict(
                    train_data="student_performance_factors_train_data",
                    label_column="params:label_column"
                ),
                outputs="autogluon_tabular_nn_model",
                name="train_tabular_nn_model_node",
            ),
            node(
                func=train_multimodal_model,
                inputs=dict(
                    train_data="student_performance_factors_train_data",
                    label_column="params:label_column"
                ),
                outputs="autogluon_multimodal_model",
                name="train_multimodal_model_node",
            ),
        ]
    )
