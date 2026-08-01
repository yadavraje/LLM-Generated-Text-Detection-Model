"""
Visualization Module
LLM Generated Text Detection

Creates training and evaluation plots.
"""

import os

import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay


REPORT_DIR = "reports/figures"

os.makedirs(REPORT_DIR, exist_ok=True)

def plot_training_history(history):
    """
    Plot training accuracy and loss.
    """

    plt.figure(figsize=(8, 5))

    plt.plot(history["accuracy"], label="Training Accuracy")

    plt.plot(history["val_accuracy"], label="Validation Accuracy")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.title("Training Accuracy")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(f"{REPORT_DIR}/training_accuracy.png")

    plt.close()


    plt.figure(figsize=(8, 5))

    plt.plot(history["loss"], label="Training Loss")

    plt.plot(history["val_loss"], label="Validation Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title("Training Loss")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(f"{REPORT_DIR}/training_loss.png")

    plt.close()

def plot_confusion_matrix(confusion_matrix):
    """
    Save confusion matrix plot.
    """

    disp = ConfusionMatrixDisplay(confusion_matrix)

    disp.plot()

    plt.tight_layout()

    plt.savefig(f"{REPORT_DIR}/confusion_matrix.png")

    plt.close()