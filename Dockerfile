FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt

COPY . .

# Normalize UTF-8 BOM and line endings (CRLF -> LF) in shell entrypoint
# to avoid "exec format error" when image is built from Windows workspaces.
RUN sed -i '1s/^\xEF\xBB\xBF//' /app/start.sh && \
    sed -i 's/\r$//' /app/start.sh && \
    chmod +x /app/start.sh

EXPOSE 7860

CMD ["/app/start.sh"]
