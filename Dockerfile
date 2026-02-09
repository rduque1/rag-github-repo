FROM python:3.12-slim-bookworm

# 1. Install system dependencies
# tesseract-ocr: The OCR engine
# libgl1 & libglib2.0-0: Required by OpenCV (used by Docling)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=docker.io/astral/uv:latest /uv /uvx /bin/

ADD . /app

WORKDIR /app

RUN uv sync --frozen

EXPOSE 8501