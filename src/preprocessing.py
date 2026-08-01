"""
Text Preprocessing Module
LLM Generated Text Detection
"""

import re
import string
import pandas as pd


def clean_text(text):
    """
    Clean a single text document.
    """

    if pd.isna(text):
        return ""

    # Convert to string
    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_dataframe(df, text_column):
    """
    Apply text preprocessing to an entire dataframe.
    """

    df = df.copy()

    df[text_column] = df[text_column].apply(clean_text)

    return df