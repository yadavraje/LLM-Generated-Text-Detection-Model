from src.data_loader import load_training_data
from src.preprocessing import preprocess_dataframe

train = load_training_data()

print("="*60)
print("Before Cleaning")
print(train.iloc[0]["text"])

train = preprocess_dataframe(train, "text")

print("="*60)
print("After Cleaning")
print(train.iloc[0]["text"])