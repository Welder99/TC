# src/models/train.py
import numpy as np
import joblib
from pathlib import Path

from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import ARTIFACTS_DIR, MODEL_PATH, SCALER_PATH
from src.data.download_data import download_price_data
from src.data.preprocess import train_test_split_sequences
from src.models.lstm_model import build_lstm_model
from tensorflow import keras

def mape(y_true, y_pred) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Coleta
    df = download_price_data()
    print(f"Total de registros: {len(df)}")

    # 2) Pré-processamento
    X_train, y_train, X_test, y_test, scaler = train_test_split_sequences(df)

    print("Shapes:")
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_test :", X_test.shape, "y_test :", y_test.shape)

    # 3) Modelo
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    model.summary()

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    # 4) Treino
    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )

    # 5) Avaliação no conjunto de teste (em escala original)
    # Carregar melhor modelo salvo
    best_model = keras.models.load_model(MODEL_PATH)

    y_pred_scaled = best_model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1))

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mape_val = mape(y_true, y_pred)

    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAPE : {mape_val:.2f}%")

    # 6) Salvar scaler
    joblib.dump(scaler, SCALER_PATH)
    print(f"Modelo salvo em {MODEL_PATH}")
    print(f"Scaler salvo em {SCALER_PATH}")

if __name__ == "__main__":
    main()
