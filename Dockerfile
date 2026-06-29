# ── Stage 1: Frontend build ──────────────────────────────────
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Server ──────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY server/ ./server/

# Copy frontend build
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# Copy data files if they exist
COPY eval_results.json* training_loss.csv* ./

EXPOSE 8000

ENV USE_MOCK=true
ENV EVAL_RESULTS_PATH=./eval_results.json
ENV LOSS_LOG_PATH=./training_loss.csv

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
