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

COPY docker/olmocr/requirements.txt /tmp/requirements.txt

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

# 1) torch/vision은 여기서 깔지 말고
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# 2) torch/vision을 “마지막에” 같은 인덱스에서 강제 재설치
RUN pip install --no-cache-dir --upgrade --force-reinstall \
    torch==2.9.1 torchvision==0.24.1 \
    --index-url https://download.pytorch.org/whl/cu128


COPY . .

ENTRYPOINT ["python", "-m", "src.ocr.cli"]
