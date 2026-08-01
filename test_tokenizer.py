from src.data_loader import load_training_data
from src.preprocessing import preprocess_dataframe
from src.tokenizer_utils import (
    create_tokenizer,
    fit_tokenizer,
    text_to_sequence,
    pad_text_sequences,
)

from src.config import TEXT_COLUMN

# Load data
train = load_training_data()

# Preprocess
train = preprocess_dataframe(train, TEXT_COLUMN)

# Create tokenizer
tokenizer = create_tokenizer()

# Fit tokenizer
tokenizer = fit_tokenizer(tokenizer, train[TEXT_COLUMN])

# Convert text to sequences
sequences = text_to_sequence(tokenizer, train[TEXT_COLUMN])

# Pad sequences
padded = pad_text_sequences(sequences)

print("Vocabulary Size:", len(tokenizer.word_index))
print("Total Samples :", len(sequences))
print("Padded Shape :", padded.shape)

print("\nFirst Sequence:")
print(sequences[0][:20])

print("\nFirst Padded Sequence:")
print(padded[0][:20])
