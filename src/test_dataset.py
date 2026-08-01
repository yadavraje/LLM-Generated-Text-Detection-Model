from src.data_loader import load_training_data
from src.dataset import prepare_dataset

df = load_training_data()

dataset = prepare_dataset(df)

X_train = dataset["X_train"]
X_test = dataset["X_test"]
y_train = dataset["y_train"]
y_test = dataset["y_test"]
tokenizer = dataset["tokenizer"]

print("=" * 50)

print("Training Samples :", len(X_train))
print("Testing Samples  :", len(X_test))
print("Vocabulary Size  :", len(tokenizer.word_index))
print("Sequence Length  :", len(X_train[0]))

print("=" * 50)