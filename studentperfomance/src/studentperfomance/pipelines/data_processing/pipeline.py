from kedro.pipeline import Pipeline, node
from .nodes import (
    preprocess_student_performance_factors,
    generate_heatmap_with_interpretation,
    generate_heatmap_encoded_with_interpretation,
    generate_correlation_plot_attendance_with_interpretation,
    generate_correlation_plot_hours_studied_with_interpretation,
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
                func=generate_heatmap_with_interpretation,
                inputs="student_performance_factors_preprocessed",
                outputs="heatmap_with_interpretation",  
                name="generate_heatmap_with_interpretation_node",
            ),
            node(
                func=generate_heatmap_encoded_with_interpretation,
                inputs="student_performance_factors_preprocessed",
                outputs="heatmap_encoded_with_interpretation",  
                name="generate_heatmap_encoded_with_interpretation_node",
            ),
            node(
                func=generate_correlation_plot_attendance_with_interpretation,
                inputs="student_performance_factors_preprocessed",
                outputs="attendance_exam_corr_plot_with_interpretation",  
                name="generate_correlation_plot_attendance_with_interpretation_node",
            ),
            node(
                func=generate_correlation_plot_hours_studied_with_interpretation,
                inputs="student_performance_factors_preprocessed",
                outputs="hours_studied_exam_corr_plot_with_intepretation",
                name="generate_correlation_plot_hours_studied_with_interpretation_node",
            ),
        ]
    )
