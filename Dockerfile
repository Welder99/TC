FROM python:3.12-slim

# Evitar buffer no stdout/stderr
ENV PYTHONUNBUFFERED=1

# Instalar dependência básica do sistema (opcional, mas ajuda algumas libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

# Instalar uv
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copiar arquivos de configuração do projeto
COPY pyproject.toml uv.lock ./

# Instalar dependências do projeto (sem dependências de dev)
RUN uv sync --no-dev --frozen

# Copiar o código fonte
COPY src ./src

# Copiar os artefatos treinados (modelo + scaler)
COPY artifacts ./artifacts

# Expor porta da API
EXPOSE 8000

# Comando para subir a API
CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
