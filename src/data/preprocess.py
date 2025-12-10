# src/data/preprocess.py
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config import WINDOW_SIZE, TEST_RATIO

def create_sequences(data: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    data: array 2D (n, 1) já escalonado
    retorna X (amostras, window_size, 1) e y (amostras,)
    """
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size : i, 0])
        y.append(data[i, 0])
    X = np.array(X)
    y = np.array(y)
    # reshape para LSTM: (amostras, passos, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y

def train_test_split_sequences(
    df: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    test_ratio: float = TEST_RATIO,
):
    """
    Divide série em treino/teste respeitando o tempo, ajusta scaler e cria sequências.
    """
    values = df["close"].values.reshape(-1, 1)

    train_size = int(len(values) * (1 - test_ratio))

    train_prices = values[:train_size]
    # Para testar, pegamos um pouco antes para montar janelas completas
    test_prices = values[train_size - window_size :]

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_prices)
    test_scaled = scaler.transform(test_prices)

    X_train, y_train = create_sequences(train_scaled, window_size)
    X_test, y_test = create_sequences(test_scaled, window_size)

    return X_train, y_train, X_test, y_test, scaler
