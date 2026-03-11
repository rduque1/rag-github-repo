FROM python:3.12-slim-bookworm

# 1. Install system dependencies
# tesseract-ocr: The OCR engine
# libgl1 & libglib2.0-0: Required by OpenCV (used by Docling)
# Playwright dependencies for headless browser
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    # Playwright/Chromium dependencies
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpango-1.0-0 \
    libcairo2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=docker.io/astral/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy only dependency files first for layer caching
COPY pyproject.toml uv.lock ./

RUN uv sync --frozen

# Install Playwright browsers
RUN uv run playwright install chromium

# Now copy the rest of the source code
COPY . .

EXPOSE 8501