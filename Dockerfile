FROM python:3.13-slim
# xgboost needs the OpenMP runtime, which slim images don't include
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY src/serve.py src/monitor.py src/scorer.py ./
COPY model/ model/
ENV MODEL_DIR=/app/model REQUEST_LOG=/tmp/requests.jsonl MLFLOW_DISABLE_AGENT_HINT=1
# Cloud Run sets PORT; default 8080 for local runs
CMD exec uvicorn serve:app --host 0.0.0.0 --port ${PORT:-8080}
