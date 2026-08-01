from src.model import build_lstm_model
from src.config import MAX_SEQUENCE_LENGTH

model = build_lstm_model()
model.build(input_shape=(None, MAX_SEQUENCE_LENGTH))
model.summary()