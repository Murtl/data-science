import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def calculate_accuracies(model_1_predictions: pd.DataFrame, 
                         model_2_predictions: pd.DataFrame, 
                         model_3_predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the accuarcy of each model based on the predictions on the test_data.
    """
    accuracies = {
        "Model": [],
        "Accuracy": []
    }
    
    for i, (predictions, model_name) in enumerate(zip(
        [model_1_predictions, model_2_predictions, model_3_predictions],
        ["Tabular Predictor", "Tabular Predictor NN", "Mutli Modal"]
    )):
        acc = accuracy_score(predictions["Actual_Score"], predictions["Pred_Score"])
        accuracies["Model"].append(model_name)
        accuracies["Accuracy"].append(acc)
    
    return pd.DataFrame(accuracies)

def visualize_accuracies(model_accuracies: pd.DataFrame) -> str:
    """
    Visualisiert die Genauigkeiten der Modelle und speichert die Grafik.
    Visualizes the Accuracy of each model and saves a picture
    """
    plt.figure(figsize=(10, 6))
    plt.bar(model_accuracies["Model"], model_accuracies["Accuracy"])
    plt.title("Model Accuracies")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    
    return plt
