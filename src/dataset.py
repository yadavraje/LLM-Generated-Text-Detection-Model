"""
dataset.py
-----------

Prepares the final training and testing datasets
for the LLM Generated Text Detection model.
"""

from sklearn.model_selection import train_test_split
from src.tokenizer_utils import (
    create_tokenizer,
    fit_tokenizer,
    text_to_sequence,
    pad_text_sequences,
)
from src.preprocessing import clean_text


from src.config import (
    TEST_SIZE,
    RANDOM_STATE,
    MAX_SEQUENCE_LENGTH
)


def prepare_dataset(df):
    """
    Prepare dataframe for model training.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    X_train
    X_test
    y_train
    y_test
    tokenizer
    """

    texts = df["text"].astype(str).apply(clean_text)

    labels = df["generated"]

    tokenizer = create_tokenizer()

    fit_tokenizer(tokenizer, texts)

    sequences = text_to_sequence(
    tokenizer,
    texts
    )

    sequences = pad_text_sequences(
    sequences
    )

    X_train, X_test, y_train, y_test = train_test_split(
    sequences,
    labels,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=labels
    )

    return {
    "X_train": X_train,
    "X_test": X_test,
    "y_train": y_train,
    "y_test": y_test,
    "tokenizer": tokenizer,
    }