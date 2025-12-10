from typing import List

import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from prometheus_fastapi_instrumentator import Instrumentator
from tensorflow import keras

from src.config import MODEL_PATH, SCALER_PATH, WINDOW_SIZE

app = FastAPI(
    title="Stock Price LSTM API",
    version="0.1.0",
    description="API para previsão de preço de fechamento usando LSTM.",
)

# Carregar modelo e scaler na importação do módulo
model = keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


class PriceHistory(BaseModel):
    closes: List[float]


class PredictionResponse(BaseModel):
    next_close: float


# Instrumentar o app ANTES de ele iniciar
instrumentator = Instrumentator().instrument(app)


@app.on_event("startup")
async def _startup():
    # Aqui apenas expomos a rota /metrics
    instrumentator.expose(app)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(history: PriceHistory):
    if len(history.closes) < WINDOW_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"É necessário pelo menos {WINDOW_SIZE} preços de fechamento.",
        )

    recent = np.array(history.closes[-WINDOW_SIZE:]).reshape(-1, 1)
    scaled = scaler.transform(recent)
    X = scaled.reshape(1, WINDOW_SIZE, 1)

    pred_scaled = model.predict(X)[0][0]
    pred = scaler.inverse_transform(np.array([[pred_scaled]]))[0][0]

    return PredictionResponse(next_close=float(pred))
