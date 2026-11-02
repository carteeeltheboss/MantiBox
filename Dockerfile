# Multi-stage Dockerfile for MaintiDoc (SPEC §3, §11)
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies: tesseract OCR + french language data, curl, build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    tesseract-ocr \
    tesseract-ocr-fra \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install project dependencies
COPY pyproject.toml README.md ./
RUN pip install --upgrade pip && \
    pip install -e .

# Copy application source code and resources
COPY src/ src/
COPY prompts/ prompts/
COPY docs/ docs/

EXPOSE 8000 8501

CMD ["uvicorn", "maintidoc.api.app:create_app_from_env", "--host", "0.0.0.0", "--port", "8000"]
