from kedro.pipeline import Pipeline, node, pipeline
from .nodes import split_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs="student_performance_factors_preprocessed",
                outputs=dict(
                    train="student_performance_factors_train_data",
                    test="student_performance_factors_test_data"
                ),
                name="split_data_node"
            )
        ]
    )
