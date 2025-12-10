# src/models/example_predict.py
import numpy as np
from tensorflow import keras
import joblib

from src.config import MODEL_PATH, SCALER_PATH, WINDOW_SIZE
from src.data.download_data import download_price_data

def main():
    # 1) carrega dados históricos
    df = download_price_data()
    closes = df["close"].values

    print("\n=== ÚLTIMOS 10 PREÇOS REAIS ===")
    print(closes[-10:])

    # 2) carrega modelo e scaler treinados
    print("\nCarregando modelo e scaler...")
    model = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # 3) pega a última janela de WINDOW_SIZE dias
    history = closes[-WINDOW_SIZE:]
    print(f"\nUsando os últimos {WINDOW_SIZE} preços para prever o próximo:")

    # 4) prepara para o modelo (escala + reshape)
    history_scaled = scaler.transform(history.reshape(-1, 1))
    X = history_scaled.reshape(1, WINDOW_SIZE, 1)

    # 5) faz a previsão
    pred_scaled = model.predict(X)[0][0]
    pred = scaler.inverse_transform([[pred_scaled]])[0][0]

    print("\n=== RESULTADO DA PREVISÃO ===")
    print(f"Próximo preço de fechamento previsto: {pred:.4f}")

    print("\nObs.: esse próximo preço é o 'dia seguinte' ao último da base baixada.")

if __name__ == "__main__":
    main()
