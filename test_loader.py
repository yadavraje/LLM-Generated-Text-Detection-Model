from src.data_loader import (
    load_training_data,
    load_testing_data,
    load_prompt_data,
    load_ai_dataset,
    dataset_summary
)

train = load_training_data()
test = load_testing_data()
prompts = load_prompt_data()
ai = load_ai_dataset()

print("Training Dataset")
dataset_summary(train)

print("\nTesting Dataset")
dataset_summary(test)

print("\nPrompt Dataset")
dataset_summary(prompts)

print("\nAI Dataset")
dataset_summary(ai)