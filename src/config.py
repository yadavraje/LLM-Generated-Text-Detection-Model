"""
Project Configuration
LLM Generated Text Detection
"""

from pathlib import Path

# =====================================================
# Project Paths
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TRAIN_DATA = DATA_DIR / "train_essays.csv"
TEST_DATA = DATA_DIR / "test_essays.csv"
PROMPT_DATA = DATA_DIR / "train_prompts.csv"
AI_DATA = DATA_DIR / "AI ESSAY.xlsx"

MODEL_PATH = MODEL_DIR / "model.h5"

# =====================================================
# Random State
# =====================================================

RANDOM_STATE = 42

# =====================================================
# Tokenizer
# =====================================================

VOCAB_SIZE = 20000

MAX_SEQUENCE_LENGTH = 500

OOV_TOKEN = "<OOV>"

# =====================================================
# Model Parameters
# =====================================================

EMBEDDING_DIM = 128

LSTM_UNITS = 128

DROPOUT = 0.3

# =====================================================
# Training Parameters
# =====================================================

EPOCHS = 10

BATCH_SIZE = 32

VALIDATION_SPLIT = 0.2

LEARNING_RATE = 0.001

MODEL_PATH = "models/model.keras"

TOKENIZER_PATH = "models/tokenizer.pkl"

HISTORY_PATH = "models/history.pkl"

# =====================================================
# Dataset Columns
# =====================================================

TEXT_COLUMN = "text"

TARGET_COLUMN = "generated"

TOKENIZER_PATH = MODEL_DIR / "tokenizer.pkl"

# Train/Test Split
TEST_SIZE = 0.20

# ==========================================
# Model Hyperparameters
# ==========================================

EMBEDDING_DIM = 128

LSTM_UNITS = 128

DENSE_UNITS = 64

DROPOUT_RATE = 0.30

OUTPUT_UNITS = 1

OUTPUT_ACTIVATION = "sigmoid"

LOSS_FUNCTION = "binary_crossentropy"

OPTIMIZER = "adam"

METRICS = ["accuracy"]
