"""
Training Module
LLM Generated Text Detection
"""

import os
import pickle

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from src.data_loader import load_training_data
from src.dataset import prepare_dataset
from src.model import build_lstm_model
from src.tokenizer_utils import save_tokenizer

from src.config import (
    BATCH_SIZE,
    EPOCHS,
    VALIDATION_SPLIT,
    MODEL_PATH,
    TOKENIZER_PATH,
    HISTORY_PATH,
)


def train_model():
    """
    Train the LSTM model and save all artifacts.
    """

    print("Loading dataset...")

    df = load_training_data()

    print("Preparing dataset...")

    dataset = prepare_dataset(df)

    X_train = dataset["X_train"]
    X_test = dataset["X_test"]
    y_train = dataset["y_train"]
    y_test = dataset["y_test"]
    tokenizer = dataset["tokenizer"]

    print("Building model...")

    model = build_lstm_model()

    os.makedirs("models", exist_ok=True)

    callbacks = [

        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),

        ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
        ),
    ]

    print("Training model...")

    history = model.fit(

        X_train,
        y_train,

        validation_split=VALIDATION_SPLIT,

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        callbacks=callbacks,

        verbose=1,
    )

    print("Saving tokenizer...")

    save_tokenizer(
        tokenizer,
        TOKENIZER_PATH,
    )

    print("Saving training history...")

    with open(HISTORY_PATH, "wb") as file:
        pickle.dump(history.history, file)

    print("Training Completed Successfully.")

    return model, history, X_test, y_test