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

def generate_heatmap_with_interpretation(student_performance_factors: pd.DataFrame):
    """
    Generates a heatmap showing the correlation between all numerical features,
    appends the interpretation to the figure, and saves the plot as an image.

    Args:
        student_performance_factors: Preprocessed data.

    Returns:
        The file path of the saved heatmap with interpretation and the Matplotlib figure.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Select only numeric columns
    numeric_data = student_performance_factors.select_dtypes(include=["number"])
       
    # Compute correlation matrix
    correlation_matrix = numeric_data.corr()
    
    # Create a figure with two parts: one for the heatmap, one for the interpretation
    fig, axs = plt.subplots(
        nrows=2, 
        ncols=1, 
        figsize=(10, 12),  # Adjust size to accommodate both plots
        gridspec_kw={'height_ratios': [3, 1]}  # Larger area for the heatmap
    )

    # Plot the heatmap
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        ax=axs[0]
    )
    axs[0].set_title("Correlation Heatmap", fontsize=16)
    axs[0].set_xlabel("")
    axs[0].set_ylabel("")

    # Generate interpretation text
    heatmap_interpretation = """
    Interpretation of the heatmap:
    - High positive correlations between Hours_Studied and Exam_Score -> 0.445.
    - High positive correlations between Attendance and Exam_Score -> 0.580.
    - The other correlations are really low or even negative (e.g. Sleep_Hours).
    """

    # Add interpretation as a text box below the heatmap
    axs[1].axis("off")  # Turn off the axes for the interpretation
    axs[1].text(
        0, 1, 
        heatmap_interpretation, 
        fontsize=12, 
        ha="left", 
        va="top", 
        wrap=True
    )

    plt.tight_layout()

    return plt


def generate_correlation_plot_attendance_with_interpretation(student_performance_factors: pd.DataFrame):
    """
    Generates a correlation plot for attendance and exam score, appends an interpretation,
    and saves the combined figure as an image.

    Args:
        student_performance_factors: Preprocessed data.

    Returns:
        The file path of the saved figure and the Matplotlib figure.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create figure with two subplots
    fig, axs = plt.subplots(
        nrows=2, 
        ncols=1, 
        figsize=(8, 10), 
        gridspec_kw={'height_ratios': [3, 1]}
    )

    # Scatter plot
    sns.scatterplot(
        x=student_performance_factors["Attendance"],
        y=student_performance_factors["Exam_Score"],
        ax=axs[0]
    )
    axs[0].set_title("Attendance vs. Exam Score")
    axs[0].set_xlabel("Attendance")
    axs[0].set_ylabel("Exam Score")

    # Interpretation
    attendance_exam_score_corr_interpretation = """
    Interpretation of the Attendance and Exam_Score correlation plot:
    - Shows a light trend in which students who participated more (higher Attendance)
      in the course achieved a higher Exam_Score.
    """
    axs[1].axis("off")
    axs[1].text(0, 1, attendance_exam_score_corr_interpretation, fontsize=12, ha="left", va="top", wrap=True)

    plt.tight_layout()
    
    return plt


def generate_correlation_plot_hours_studied_with_interpretation(student_performance_factors: pd.DataFrame):
    """
    Generates a correlation plot for hours studied and exam score, appends an interpretation,
    and saves the combined figure as an image.

    Args:
        student_performance_factors: Preprocessed data.

    Returns:
        The file path of the saved figure and the Matplotlib figure.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create figure with two subplots
    fig, axs = plt.subplots(
        nrows=2, 
        ncols=1, 
        figsize=(8, 10), 
        gridspec_kw={'height_ratios': [3, 1]}
    )

    # Scatter plot
    sns.scatterplot(
        x=student_performance_factors["Hours_Studied"],
        y=student_performance_factors["Exam_Score"],
        ax=axs[0]
    )
    axs[0].set_title("Hours Studied vs. Exam Score")
    axs[0].set_xlabel("Hours Studied")
    axs[0].set_ylabel("Exam Score")

    # Interpretation
    hours_studied_exam_score_corr_interpretation = """
    Interpretation of the Hours_Studied and Exam_Score correlation plot:
    - Shows a trend in which students who studied more hours (Hours_Studied) achieved a higher Exam_Score.
    """
    axs[1].axis("off")
    axs[1].text(0, 1, hours_studied_exam_score_corr_interpretation, fontsize=12, ha="left", va="top", wrap=True)

    plt.tight_layout()

    return plt


def encode_categorical_columns(student_performance_factors: pd.DataFrame) -> pd.DataFrame:
    """Encodes categorical columns using custom mappings.

    Args:
        student_performance_factors: Original data with categorical columns.
        encodings: A dictionary of column names and their corresponding encodings.

    Returns:
        DataFrame with encoded categorical columns.
    """
    # Custom encodings
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
    # Copy the input DataFrame to avoid modifying the original
    encoded_data = student_performance_factors.copy()
    
    # Apply encodings
    for column, mapping in custom_encodings.items():
        if column in encoded_data.columns:
            encoded_data[column] = encoded_data[column].map({val: idx for idx, val in enumerate(mapping)})
    
    return encoded_data


def generate_heatmap_encoded_with_interpretation(student_performance_factors: pd.DataFrame):
    """
    Generates a heatmap showing the correlation between all numerical features including encoded
    categorical columns, appends an interpretation, and saves the combined figure as an image.

    Args:
        student_performance_factors: Preprocessed data.

    Returns:
        The file path of the saved figure and the Matplotlib figure.
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
    correlation_matrix = numeric_data.corr()

    # Create figure with two subplots
    fig, axs = plt.subplots(
        nrows=2, 
        ncols=1, 
        figsize=(12, 14), 
        gridspec_kw={'height_ratios': [3, 1]}
    )

    # Heatmap
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        ax=axs[0]
    )
    axs[0].set_title("Correlation Heatmap with Encoded Data", fontsize=16)

    # Interpretation
    heatmap_encoded_interpretation = """
    Interpretation of the encoded heatmap:
    - High positive correlations between Hours_Studied and Exam_Score -> 0.445.
    - High positive correlations between Attendance and Exam_Score -> 0.580.
    - The other correlations are really low or even negative (e.g. Sleep_Hours).
    - The encoded heatmap therefore shows nothing new compared to the other heatmap,
      where categorical values were not taken into account.
    """
    axs[1].axis("off")
    axs[1].text(0, 1, heatmap_encoded_interpretation, fontsize=12, ha="left", va="top", wrap=True)

    plt.tight_layout()

    return plt
