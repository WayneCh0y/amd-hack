# Track 1 agent image. Judging VM runs linux/amd64 — build with:
#   docker buildx build --platform linux/amd64 -t <registry>/amd-track1:latest --push .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application source only (see .dockerignore for what is excluded).
COPY src/ ./src/

ENV PYTHONPATH=/app/src

# Read tasks from /input, write results to /output (created if absent at runtime).
ENTRYPOINT ["python", "-m", "agent.main"]
