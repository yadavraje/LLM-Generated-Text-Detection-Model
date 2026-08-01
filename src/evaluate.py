"""
Evaluation Module
LLM Generated Text Detection

Evaluates the trained LSTM model.
"""
from src.visualization import plot_confusion_matrix
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.train import train_model


def evaluate_model():
    """
    Train and evaluate the LSTM model.
    """

    model, history, X_test, y_test = train_model()

    print("\nEvaluating model...\n")

    predictions = model.predict(X_test)

    predictions = (predictions > 0.5).astype(int).flatten()

    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy : {accuracy:.4f}")

    print("\nClassification Report\n")

    print(classification_report(y_test, predictions))

    cm = confusion_matrix(y_test, predictions)

    print("\nConfusion Matrix\n")

    print(cm)

    plot_confusion_matrix(cm)

    print("\nConfusion matrix image saved successfully.")

    return {
    "accuracy": accuracy,
    "predictions": predictions,
    "actual": y_test,
    }