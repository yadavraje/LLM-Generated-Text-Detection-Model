from tensorflow.keras.layers import Input
from src.config import (
    VOCAB_SIZE,
    MAX_SEQUENCE_LENGTH,
    EMBEDDING_DIM,
    LSTM_UNITS,
    DENSE_UNITS,
    DROPOUT_RATE,
    OUTPUT_UNITS,
    OUTPUT_ACTIVATION,
    LOSS_FUNCTION,
    OPTIMIZER,
    METRICS,
)

"""
Model Module
LLM Generated Text Detection

Builds and compiles the LSTM model.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Dense,
    Dropout,
)


def build_lstm_model():
    """
    Build and compile the LSTM model.
    """

    model = Sequential()
    model.add(Input(shape=(MAX_SEQUENCE_LENGTH,)))
    model.add(
    Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMBEDDING_DIM,
        )
    )


    model.add(LSTM(LSTM_UNITS))

    model.add(
        Dropout(
            DROPOUT_RATE
        )
    )

    model.add(
        Dense(
            DENSE_UNITS,
            activation="relu",
        )
    )

    model.add(
        Dropout(
            DROPOUT_RATE
        )
    )

    model.add(
        Dense(
            OUTPUT_UNITS,
            activation=OUTPUT_ACTIVATION,
        )
    )

    model.compile(
        optimizer=OPTIMIZER,
        loss=LOSS_FUNCTION,
        metrics=METRICS,
    )

    return model