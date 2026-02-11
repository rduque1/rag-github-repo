FROM python:3.11-slim

WORKDIR /app

# Install common Python packages for data science and general use
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    matplotlib \
    openpyxl \
    requests \
    httpx \
    pydantic \
    python-dateutil \
    flask

COPY server.py .

# Create non-root user and switch to it for security
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

EXPOSE 5000

CMD ["python", "server.py"]
