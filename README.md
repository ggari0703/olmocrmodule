## 개요

이 저장소는 여러 OCR 엔진(olmOCR VLM, TrOCR, Donut)을 단일 CLI로 묶어 제공합니다. 모델마다 독립적인 Dockerfile과 `requirements.txt`를 사용하므로, 가상환경과 패키지 버전을 엔진별로 분리해 관리할 수 있습니다.

```
.
├── docker/
│   ├── donut/
│   ├── olmocr/
│   └── trocr/
├── docker-compose.yml
├── model_downloads.py
├── models/                  # Hugging Face 스냅샷 위치
└── src/ocr/
    ├── cli.py               # python -m src.ocr.cli
    ├── engines/
    └── pipeline.py
```

## 모델 준비

1. `models/` 디렉터리를 만들고 필요한 모델 스냅샷을 받아주세요. 제공된 스크립트는 `HF_TOKEN` 환경변수에 Hugging Face 토큰이 있어야 합니다.

   ```bash
   export HF_TOKEN=hf_xxx
   uv pip install huggingface_hub
   python model_downloads.py
   ```

2. 기본적으로 CLI는 아래 폴더들을 찾습니다.

   - `models/allenai__olmOCR-2-7B-1025-FP8`
   - `models/Qwen__Qwen2.5-VL-7B-Instruct`
   - `models/microsoft__trocr-base-printed` (선택)
   - `models/naver-clova-ix__donut-base` (선택)

   다른 폴더명을 쓰고 싶다면 `src/ocr/cli.py`의 경로를 수정하세요.

## Docker 기반 엔진 실행

각 엔진은 고유한 Dockerfile/요구사항 파일을 가지며, 공통 소스(`/app`)를 공유하지만 필요한 패키지만 설치합니다.

### 이미지 빌드

```bash
# 전체 빌드
docker compose build

# 개별 빌드
docker compose build olmocr
docker compose build trocr
docker compose build donut
```

Compose는 모든 컨테이너에 다음 볼륨을 연결합니다.

- `./models` → `/app/models` (오프라인 모델 가중치)
- `./data` → `/app/data` (PDF 입력)
- `./output` → `/app/output` (결과 텍스트)

없다면 `data/`, `output/` 디렉터리를 만들어 주세요.

### Docker Compose로 OCR 실행

서비스마다 `--engine` 값이 이미 지정되어 있어 나머지 CLI 인자만 넘기면 됩니다. 예시(olmOCR VLM):

```bash
docker compose run --rm olmocr \
  --pdf /app/data/sample.pdf \
  --page_end 1 \
  --out /app/output/sample.txt \
  --models_dir /app/models
```

다른 엔진을 쓰려면 서비스 이름만 바꾸면 됩니다.

```bash
docker compose run --rm trocr --pdf /app/data/... --page_end 3 --out /app/output/trocr.txt --models_dir /app/models
docker compose run --rm donut --pdf /app/data/... --page_end 2 --out /app/output/donut.txt --models_dir /app/models
```

### `docker run`으로 직접 실행

Compose 대신 Dockerfile을 직접 빌드/실행해도 됩니다.

```bash
docker build -f docker/olmocr/Dockerfile -t ocr-olmocr .
docker run --rm \
  -v "$PWD/models":/app/models \
  -v "$PWD/data":/app/data \
  -v "$PWD/output":/app/output \
  ocr-olmocr \
  --pdf /app/data/sample.pdf --page_end 1 --out /app/output/out.txt --models_dir /app/models
```

다른 엔진도 동일한 절차로 실행합니다.

## 로컬 CLI 실행(선택)

Docker 없이 실행하려면 원하는 엔진의 요구사항 파일을 설치하고 다음처럼 CLI를 호출하세요.

```bash
uv pip install -r docker/olmocr/requirements.txt  # trocr/donut도 동일
python -m src.ocr.cli --pdf ./tests/sample.pdf --page_end 1 --models_dir ./models
```

각 엔진 구현은 `src/ocr/engines/`에 있으며, 추상 `OCREngine` 인터페이스 덕분에 서로 영향을 주지 않고 교체/추가할 수 있습니다.
