from kedro.pipeline import Pipeline, node
from .nodes import (
    preprocess_student_performance_factors,
    generate_heatmap,
    generate_heatmap_encoded,
    generate_correlation_plot_attendance,
    generate_correlation_plot_hours_studied,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=preprocess_student_performance_factors,
                inputs="student_performance_factors",
                outputs="student_performance_factors_preprocessed",
                name="preprocess_student_performance_factors_node",
            ),
            node(
                func=generate_heatmap,
                inputs="student_performance_factors_preprocessed",
                outputs="heatmap",  
                name="generate_heatmap_node",
            ),
            node(
                func=generate_heatmap_encoded,
                inputs="student_performance_factors_preprocessed",
                outputs="heatmap_encoded",  
                name="generate_heatmap_encoded_node",
            ),
            node(
                func=generate_correlation_plot_attendance,
                inputs="student_performance_factors_preprocessed",
                outputs="attendance_exam_corr_plot",  
                name="generate_correlation_plot_attendance_node",
            ),
            node(
                func=generate_correlation_plot_hours_studied,
                inputs="student_performance_factors_preprocessed",
                outputs="hours_studied_exam_corr_plot",
                name="generate_correlation_plot_hours_studied_node",
            ),
        ]
    )
