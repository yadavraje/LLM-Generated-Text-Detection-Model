from src.train import train_model
from src.visualization import plot_training_history

model, history, X_test, y_test = train_model()

plot_training_history(history.history)

print("Training plots saved successfully.")