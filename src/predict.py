"""
Prediction Module
LLM Generated Text Detection
"""
from utils.model_loader import load_prediction_resources

from tensorflow.keras.models import load_model

from src.tokenizer_utils import (
    load_tokenizer,
    text_to_sequence,
    pad_text_sequences,
)

from src.preprocessing import clean_text

from src.config import (
    MODEL_PATH,
    TOKENIZER_PATH,
)


def predict_text(text):
    """
    Predict whether the supplied text is AI-generated or Human-written.
    """

    model, tokenizer = load_prediction_resources()

    cleaned_text = clean_text(text)

    sequence = text_to_sequence(
        tokenizer,
        [cleaned_text],
    )

    sequence = pad_text_sequences(sequence)

    probability = model.predict(sequence, verbose=0)[0][0]

    prediction = "AI Generated" if probability >= 0.5 else "Human Written"

    return {
        "prediction": prediction,
        "confidence": float(probability),
    }