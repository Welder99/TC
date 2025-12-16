# src/config.py
from pathlib import Path

# Símbolo da ação – troque para a empresa que você vai usar no trabalho
SYMBOL = "PETR4.SA"  # Ex.: "PETR4.SA" para Petrobras na B3

START_DATE = "2018-01-01"
END_DATE = "2025-12-09"

WINDOW_SIZE = 5       
TEST_RATIO = 0.2       

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "lstm_stock.keras"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
