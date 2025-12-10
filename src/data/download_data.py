# src/data/download_data.py
import yfinance as yf
import pandas as pd
from src.config import SYMBOL, START_DATE, END_DATE

def download_price_data(
    symbol: str = SYMBOL,
    start: str = START_DATE,
    end: str = END_DATE,
) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end)
    if df.empty:
        raise ValueError("Nenhum dado retornado. Verifique o símbolo e o intervalo de datas.")
    df = df[["Close"]].dropna()
    df.rename(columns={"Close": "close"}, inplace=True)
    return df


if __name__ == "__main__":
    df = download_price_data()

    print("\n=== PRIMEIRAS 5 LINHAS (início da série) ===")
    print(df.head())

    print("\n=== ÚLTIMAS 5 LINHAS (preços mais recentes) ===")
    print(df.tail())

    print("\nTotal de registros:", len(df))