# src/models/lstm_model.py
from tensorflow import keras
from tensorflow.keras import layers

def build_lstm_model(input_shape):
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(1),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model
