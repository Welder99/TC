# Tech Challenge – LSTM para Previsão de Preço de Ações

Projeto desenvolvido para o **Tech Challenge – Fase 4 (Deep Learning e IA)**, com o objetivo de:

> Criar um modelo preditivo utilizando **redes neurais LSTM (Long Short-Term Memory)** para prever o **valor de fechamento** de ações de uma empresa, realizando **toda a pipeline**:  
> coleta de dados, pré-processamento, treinamento, avaliação, salvamento do modelo, deploy em uma **API REST** e configuração de monitoramento básico.

---

## 🧠 Visão Geral do Projeto

Este projeto:

1. **Coleta dados históricos de preços de ações** via [Yahoo Finance](https://finance.yahoo.com/) usando a biblioteca `yfinance`.
2. **Pré-processa a série temporal** (normalização, janelas deslizantes, divisão treino/teste).
3. **Treina um modelo LSTM** em Python (TensorFlow/Keras).
4. **Avalia o modelo** usando métricas como **MAE**, **RMSE** e **MAPE**.
5. **Salva o modelo treinado** e o scaler de normalização em disco.
6. **Expõe uma API REST** (FastAPI) que:
   - recebe uma lista de **preços históricos de fechamento**;
   - retorna a **previsão do próximo preço de fechamento**.
7. **Expõe métricas de monitoramento** no formato Prometheus em `/metrics`.
8. Possui **Dockerfile** para facilitar o deploy em nuvem.

---

## 🏗 Arquitetura do Projeto

Estrutura de pastas (simplificada):

```txt
.
├── pyproject.toml
├── uv.lock
├── Dockerfile
├── artifacts/
│   ├── lstm_stock.keras      # modelo LSTM treinado
│   └── scaler.pkl            # scaler de normalização (MinMaxScaler)
└── src/
    ├── __init__.py
    ├── config.py             # configurações gerais (símbolo, datas, paths etc.)
    ├── data/
    │   ├── __init__.py
    │   ├── download_data.py  # coleta dados do Yahoo Finance
    │   └── preprocess.py     # pré-processamento para LSTM
    ├── models/
    │   ├── __init__.py
    │   ├── lstm_model.py     # definição da arquitetura LSTM
    │   ├── train.py          # script de treinamento e avaliação
    │   └── example_predict.py# exemplo de previsão direta no console
    └── api/
        ├── __init__.py
        └── main.py           # API FastAPI (endpoints /health, /predict, /metrics)


🧰 Tecnologias Utilizadas

Linguagem: Python 3.12

Gerenciador de ambiente/dependências: uv

Coleta de dados: yfinance, pandas

Machine Learning / Deep Learning: TensorFlow, Keras, scikit-learn, numpy

API REST: FastAPI, Uvicorn

Monitoramento: prometheus-fastapi-instrumentator (exposição de métricas em /metrics)

Containerização: Docker

⚙️ Configuração do Ambiente
1. Clonar o repositório
git clone <URL_DO_REPOSITORIO>.git
cd <NOME_DA_PASTA>

2. Instalar o uv (se ainda não tiver)
pip install uv

3. Instalar as dependências

Na raiz do projeto (onde está o pyproject.toml):

uv sync


O uv vai criar/gerenciar o ambiente virtual e instalar as libs necessárias.

Sempre que for rodar algo, use uv run ... para garantir que está usando o ambiente correto.

🔧 Configuração da Ação (Símbolo, Datas, Janela)

No arquivo src/config.py ficam as principais configurações:

from pathlib import Path

SYMBOL = "PETR4.SA"  # Símbolo da ação (ex.: "PETR4.SA", "VALE3.SA")
START_DATE = "2018-01-01"
END_DATE = "2025-12-09"

WINDOW_SIZE = 5      # quantidade de dias usados como janela de entrada da LSTM
TEST_RATIO = 0.2      # 20% dos dados para teste

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "lstm_stock.keras"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"


Você pode alterar:

SYMBOL para a empresa que o trabalho exigir.

Datas inicial/final, respeitando a disponibilidade no Yahoo Finance.

WINDOW_SIZE (tamanho da janela de entrada da LSTM).

📥 1. Coleta e Visualização dos Dados

Para ver os dados históricos da ação (primeiras/últimas linhas, estatísticas):

uv run python -m src.data.download_data


Saída esperada (exemplo):

=== PRIMEIRAS 5 LINHAS (início da série) ===
                 close
Date
2018-01-02   110.55
2018-01-03   111.13
...

=== ÚLTIMAS 5 LINHAS (preços mais recentes) ===
                 close
Date
2024-07-15   102.30
2024-07-16   103.12
...

Total de registros: 1647

=== ESTATÍSTICAS DA SÉRIE DE PREÇOS ===
Preço médio       : 112.34
Preço mínimo      : 85.12
Preço máximo      : 154.78


Essa etapa demonstra a coleta via yfinance e uma visão geral da série.

🧪 2. Treinamento do Modelo LSTM

O script de treinamento:

Baixa os dados (download_price_data).

Pré-processa (normaliza, cria janelas, separa treino/teste).

Constrói o modelo LSTM.

Treina com callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint).

Avalia nas métricas MAE, RMSE, MAPE.

Salva o modelo e o scaler em artifacts/.

Para treinar:

uv run python -m src.models.train


Ao final, o script imprime algo assim (exemplo ilustrativo):

MAE  : 1.23
RMSE : 1.85
MAPE : 2.10%
Modelo salvo em artifacts/lstm_stock.keras
Scaler salvo em artifacts/scaler.pkl

Interpretação das Métricas

MAE (Mean Absolute Error): erro médio absoluto em unidades de preço
→ ex.: MAE = 1.23 significa erro médio de R$ 1,23.

RMSE (Root Mean Square Error): similar ao MAE, mas penaliza mais erros grandes.

MAPE (Mean Absolute Percentage Error): erro percentual médio
→ ex.: MAPE = 2.10% significa erro médio de ~2,1% em relação ao valor real.

Na documentação do projeto / relatório, você pode comentar se esses valores são aceitáveis considerando a faixa de preço da ação escolhida.

🔍 3. Exemplo de Previsão Direta (sem API)

Para testar o modelo diretamente via script (usando os últimos dados históricos da ação):

uv run python -m src.models.example_predict


Saída típica:

=== ÚLTIMOS 10 PREÇOS REAIS ===
[102.30 103.12 101.80 100.90 102.70 103.40 104.00 103.60 104.20 105.10]

Carregando modelo e scaler...

Usando os últimos 60 preços para prever o próximo:
1/1 ━━━━━ 0s 30ms/step

=== RESULTADO DA PREVISÃO ===
Próximo preço de fechamento previsto: 105.87


Esse script é útil para fins de demonstração e validação rápida do modelo.

🌐 4. API REST (FastAPI)

A API está implementada em src/api/main.py.

4.1. Subir a API
uv run uvicorn src.api.main:app --reload


Se tudo estiver correto, o servidor irá rodar em:

http://127.0.0.1:8000

4.2. Endpoints Disponíveis
GET /health

Verifica se a API está saudável.

Exemplo de resposta:

{
  "status": "ok"
}

POST /predict

Recebe dados históricos de preços de fechamento e devolve a previsão do próximo preço de fechamento.

Request body (JSON):

{
  "closes": [
    100.5,
    101.2,
    99.8,
    102.3,
    103.4,
    ...
  ]
}


O campo closes é uma lista de valores float (preços de fechamento em ordem temporal).

É necessário fornecer pelo menos WINDOW_SIZE valores (configurado em config.py).

Response (JSON):

{
  "next_close": 105.8732
}


Onde next_close é o próximo preço de fechamento previsto pelo modelo LSTM.

4.3. Testando via Swagger (Interface Gráfica)

Com a API rodando, acesse:

http://127.0.0.1:8000/docs

Lá você pode:

Ver os endpoints (/health, /predict).

Clicar em "Try it out" no /predict.

Enviar um JSON com a lista de closes.

Ver a resposta da previsão na própria interface web.

4.4. Exemplo de chamada via curl
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "closes": [
    100.5, 101.2, 99.8, 102.3, 103.4, 104.0, 103.6, 104.2, 105.1, 104.8,
    105.3, 105.7, 106.1, 105.9, 106.4, 107.0, 107.5, 107.2, 107.9, 108.3,
    108.8, 108.5, 109.0, 109.4, 109.9, 110.3, 110.8, 111.2, 111.7, 112.1,
    112.6, 113.0, 113.5, 113.9, 114.4, 114.8, 115.3, 115.7, 116.2, 116.6,
    117.1, 117.5, 118.0, 118.4, 118.9, 119.3, 119.8, 120.2, 120.7, 121.1,
    121.6, 122.0, 122.5, 122.9, 123.4, 123.8, 124.3, 124.7, 125.2
  ]
}'

📊 5. Monitoramento (Escalabilidade e Observabilidade)

Para atender ao requisito de escabilidade e monitoramento, a API utiliza:

prometheus-fastapi-instrumentator

No startup da aplicação, o instrumentador:

Adiciona middleware para medir:

tempo de resposta,

contagem de requisições,

códigos de status etc.

Expõe as métricas em:

GET /metrics

5.1. Acessando métricas

Com a API rodando:

Acesse: http://127.0.0.1:8000/metrics

Você verá um texto em formato Prometheus, com várias métricas que podem ser coletadas num ambiente real por:

Servidor Prometheus

Painel Grafana

Na documentação/relatório, basta explicar que:

“Para monitorar o modelo em produção, a API expõe métricas em formato Prometheus no endpoint /metrics, permitindo monitorar tempo de resposta e volume de acessos. Em um ambiente real, essas métricas seriam coletadas por Prometheus e visualizadas em Grafana.”

🐳 6. Docker (Deploy em Container)

O projeto inclui um Dockerfile para facilitar o deploy da API em qualquer ambiente compatível com Docker (nuvem ou on-premise).

6.1. Build da imagem

Na raiz do projeto:

docker build -t lstm-stock-api .

6.2. Rodar o container
docker run -p 8000:8000 lstm-stock-api


A API ficará acessível em:

http://127.0.0.1:8000

Swagger: http://127.0.0.1:8000/docs

Métricas: http://127.0.0.1:8000/metrics

6.3. Deploy em Nuvem (opcional)

Você pode subir essa imagem em qualquer provedor de nuvem que suporte Docker, por exemplo:

Railway

Render

Fly.io

Azure Container Apps / Web Apps

Google Cloud Run

AWS ECS / Fargate

No relatório, basta incluir:

Print ou link da API rodando na nuvem (se fizer o deploy).

Breve descrição de como o container foi publicado.