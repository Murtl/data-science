import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def preprocess_student_performance_factors(student_performance_factors: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data by removing missing values -> because it does not make sense to synthesize them.

    Args:
        student_performance_factors: Raw data.
    Returns:
        Preprocessed data, without rows with missing values.
    """
    student_performance_factors.dropna(subset=["Parental_Education_Level", "Teacher_Quality", "Distance_from_Home"], inplace=True)
    return student_performance_factors

def generate_heatmap(student_performance_factors: pd.DataFrame):
    """
    Generates a heatmap showing the correlation between all numerical features,
    and saves the plot as an image.

    Args:
        student_performance_factors: Preprocessed data.

    Returns:
        The Matplotlib figure.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Select only numeric columns
    numeric_data = student_performance_factors.select_dtypes(include=["number"])
       
    # Compute correlation matrix and plot the heatmap
    plt.figure(figsize=(10,8))
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.tight_layout()

    return plt


def generate_correlation_plot_attendance(student_performance_factors: pd.DataFrame):
    """
    Generates a correlation plot for attendance and exam score,
    and saves the combined figure as an image.

    Args:
        student_performance_factors: Preprocessed data.

    Returns:
        The Matplotlib figure.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8, 6))
    
    # Scatter plot
    sns.scatterplot(
        x=student_performance_factors["Attendance"],
        y=student_performance_factors["Exam_Score"]
    )

    plt.title(f'Scatter plot of exam_score and attendance')
    plt.xlabel('Attendance')
    plt.ylabel('Exam Score')
    plt.tight_layout()
    
    return plt


def generate_correlation_plot_hours_studied(student_performance_factors: pd.DataFrame):
    """
    Generates a correlation plot for hours studied and exam score,
    and saves the combined figure as an image.

    Args:
        student_performance_factors: Preprocessed data.

    Returns:
        The Matplotlib figure.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8, 6))
    
    # Scatter plot
    sns.scatterplot(
        x=student_performance_factors["Hours_Studied"],
        y=student_performance_factors["Exam_Score"]
    )
    
    plt.title(f'Scatter plot of exam_score and attendance')
    plt.xlabel('Hours_Studied')
    plt.ylabel('Exam Score')
    plt.tight_layout()

    return plt


def generate_heatmap_encoded(student_performance_factors: pd.DataFrame):
    """
    Generates a heatmap showing the correlation between all numerical features including encoded
    categorical columns, and saves the combined figure as an image.

    Args:
        student_performance_factors: Preprocessed data.

    Returns:
        The Matplotlib figure.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Encode categorical columns
    def encode_categorical_columns(student_performance_factors):
        custom_encodings = {
            'Parental_Involvement': ['Low', 'Medium', 'High'],
            'Access_to_Resources': ['Low', 'Medium', 'High'],
            'Extracurricular_Activities': ['No', 'Yes'],
            'Motivation_Level': ['Low', 'Medium', 'High'],
            'Internet_Access': ['No', 'Yes'],
            'Family_Income': ['Low', 'Medium', 'High'],
            'School_Type': ['Public', 'Private'],
            'Peer_Influence': ['Negative', 'Neutral', 'Positive'],
            'Learning_Disabilities': ['No', 'Yes'],
            'Gender': ['Female', 'Male'],
        }
        encoded_data = student_performance_factors.copy()
        for column, mapping in custom_encodings.items():
            if column in encoded_data.columns:
                encoded_data[column] = encoded_data[column].map({val: idx for idx, val in enumerate(mapping)})
        return encoded_data

    encoded_data = encode_categorical_columns(student_performance_factors)

    # Select numeric columns and compute correlation matrix
    numeric_data = encoded_data.select_dtypes(include=["number"])

    # Compute correlation matrix and plot the heatmap
    plt.figure(figsize=(10,8))
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt=".2f", square=True, cbar_kws={"shrink": .8})
    plt.title('Correlation Heatmap with encoded categorical and bool variables')
    plt.tight_layout()
    
    return plt
