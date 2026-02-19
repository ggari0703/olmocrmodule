FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY docker/donut/requirements.txt /tmp/requirements.txt

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==2.4.1 --index-url ${TORCH_INDEX_URL} \
    && pip install --no-cache-dir -r /tmp/requirements.txt

COPY . .

ENTRYPOINT ["python", "-m", "src.ocr.cli"]
