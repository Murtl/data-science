# data-science
Repo for my data-science project (Modul: Data Science WiSe24/25)

## Student Performance Factors
Dataset from Kaggle -> https://www.kaggle.com/datasets/lainguyn123/student-performance-factors

This dataset provides a comprehensive overview of various factors affecting student performance in exams. It includes information on study habits, attendance, parental involvement, and other aspects influencing academic success.

### Research Questions
Which factors influence student performance the most, and how can we predict a student’s performance based on these factors?

### Structure
- `pre-pipeline-research/`: Contains the pre-pipeline research that is not part of the report but was necessary to understand the data, the problem and AutoGluon
  (for more information take a look into the `student_performance_factors.ipynb` storybook in the folder).
- `studentperformance/`: Contains the main research of the report and the whole kedro pipeline (see `studentperformance/visualization` for the pipeline visualization).
  The methodology described in the next sections refers to the parts of the pipeline.
- `talk-slides`: Contains the slides for the talk about the project that I gave in the course on 2024/12/12.
- `anaconda-environment.yml`: Contains the environment for the project which can be used to recreate the environment and the research.
- `Student_Performance_Report_Michael_Mertl.pdf`: Contains the final report of the project.

### Methodology

### Step 1: Data Preprocessing
- The dataset was cleaned by removing missing values and duplicates.
- In the pipeline:
  - raw data: `data/01_raw/student_performance_factors.csv`
  - cleaned data: `data/02_preprocessed/student_performance_factors_preprocessed.csv`
  - sub-pipeline with code: `src/studentperformance/pipelines/data_preprocessing/`
  - notebook: `notebooks/1_data_preprocessing.ipynb`

### Step 2: Statistical Analysis
- A correlation heatmap was created to identify the most important features.
- The heatmap shows that the most important features for the `exam_score` are `attendance` and `hours_studied`.
- The custom encoded heatmap (where all values were made numeric) shows this too.
- For these two features two extra scatter plots were created.
- In the pipeline:
  - plots: `data/08_reporting/`
  - sub-pipeline with code: `src/studentperformance/pipelines/data_preprocessing/`
  - notebook: `notebooks/1_data_preprocessing.ipynb`

### Step 3: AutoML with AutoGluon preparation
- The data was split into training and test sets.
- In the pipeline:
  - training data: `data/03_train_data/student_performance_factors_train_data.csv`
  - test data: `data/04_test_data/student_performance_factors_test_data.csv`
  - sub-pipeline with code: `src/studentperformance/pipelines/data_science_prep/`
  - notebook: `notebooks/2_data_science_prep.ipynb`

### Step 4: AutoML with AutoGluon training
- AutoGluons TabularPredictor and MultiModalPredictor were used and trained.
- The TabularPredictor was used two times, once with all models and once with focus on neural networks.
- In the pipeline:
  - models: `data/05_models/`
  - sub-pipeline with code: `src/studentperformance/pipelines/data_science_training/`
  - notebook: `notebooks/3_data_science_training.ipynb`

### Step 5: AutoML with AutoGluon prediction
- The trained models were used to predict the `exam_score` of the test data.
- In the pipeline:
  - predictions: `data/06_predictions/`
  - sub-pipeline with code: `src/studentperformance/pipelines/data_science_prediction/`
  - notebook: `notebooks/4_data_science_pred.ipynb`

### Step 6: Model Comparison
- The accuracy of the predictions of the three models was compared.
- The result was that the `WeightedEnsemble_L3` model of the TabularPredictor was the best model with an accuracy of 0.8713 (0.8777 on the test_data).
- The feature importance of the best model was also analyzed -> `attendance` (0.756) and `hours_studied` (0.734) are the most important features.
- In the pipeline:
  - plots: `data/07_model_comparison/`
  - sub-pipeline with code: `src/studentperformance/pipelines/data_science_comparison/`
  - notebook: `notebooks/5_data_science_comparison.ipynb`

### Step 7: "Baseline" Comparison
- The next step (that is not part of the pipeline but of the report) was to compare the best model to other models from other researchers.
- For this, different models from Kaggle on the same dataset were used.
- This whole analysis and final comparison is present in the report (`Student_Performance_Report_Michael_Mertl.pdf`).
- The extra calculated metric scores, besides the accuracy, are present in the `studentperformance/notebooks/5_data_science_comparison.ipynb` notebook
and are used to compare my model with the other Kaggle models.