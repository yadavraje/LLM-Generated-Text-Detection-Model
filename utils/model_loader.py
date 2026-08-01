"""
Model Loader
Loads trained model and tokenizer only once.
"""

import streamlit as st
from tensorflow.keras.models import load_model

from src.tokenizer_utils import load_tokenizer
from src.config import (
    MODEL_PATH,
    TOKENIZER_PATH,
)


@st.cache_resource
def load_prediction_resources():
    """
    Load trained model and tokenizer.

    Returns
    -------
    model
    tokenizer
    """

    model = load_model(MODEL_PATH)

    tokenizer = load_tokenizer(TOKENIZER_PATH)

    return model, tokenizer