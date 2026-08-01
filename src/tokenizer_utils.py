"""
Tokenizer Utility Module
LLM Generated Text Detection
"""

import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.config import (
    VOCAB_SIZE,
    MAX_SEQUENCE_LENGTH,
    OOV_TOKEN,
)


def create_tokenizer():
    """
    Create a Keras tokenizer.
    """
    return Tokenizer(
        num_words=VOCAB_SIZE,
        oov_token=OOV_TOKEN
    )


def fit_tokenizer(tokenizer, texts):
    """
    Fit tokenizer on training texts.
    """
    tokenizer.fit_on_texts(texts)
    return tokenizer


def text_to_sequence(tokenizer, texts):
    """
    Convert text into integer sequences.
    """
    return tokenizer.texts_to_sequences(texts)


def pad_text_sequences(sequences):
    """
    Pad sequences to equal length.
    """
    return pad_sequences(
        sequences,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post"
    )


def save_tokenizer(tokenizer, filepath):
    """
    Save tokenizer to disk.
    """
    with open(filepath, "wb") as file:
        pickle.dump(tokenizer, file)


def load_tokenizer(filepath):
    """
    Load tokenizer from disk.
    """
    with open(filepath, "rb") as file:
        tokenizer = pickle.load(file)

    return tokenizer