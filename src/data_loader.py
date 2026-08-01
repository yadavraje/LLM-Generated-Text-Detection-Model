"""
Data Loader Module
LLM Generated Text Detection
"""

from pathlib import Path
import pandas as pd

from src.config import (
    TRAIN_DATA,
    TEST_DATA,
    PROMPT_DATA,
    AI_DATA
)


def load_training_data():
    """
    Load the training essays dataset.
    """
    return pd.read_csv(TRAIN_DATA)


def load_testing_data():
    """
    Load the testing essays dataset.
    """
    return pd.read_csv(TEST_DATA)


def load_prompt_data():
    """
    Load the prompt dataset.
    """
    return pd.read_csv(PROMPT_DATA)


def load_ai_dataset():
    """
    Load the additional AI essay dataset.
    """
    return pd.read_excel(AI_DATA)


def dataset_summary(df):
    """
    Display basic dataset information.
    """
    print("=" * 50)
    print("Dataset Shape :", df.shape)
    print("=" * 50)
    print("\nColumns")
    print(df.columns.tolist())

    print("\nMissing Values")
    print(df.isnull().sum())

    print("\nData Types")
    print(df.dtypes)

    print("=" * 50)