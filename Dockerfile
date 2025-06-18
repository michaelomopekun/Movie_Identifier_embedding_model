FROM python:3.10-slim AS builder

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies into a temp dir
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --prefix=/install torch==2.2.2 --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
 && pip install --prefix=/install -r requirements.txt --no-cache-dir

# Runtime
FROM python:3.10-slim

WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /install /usr/local

# Copy app code
COPY . .

ENV PORT=8000
EXPOSE 8000

CMD ["gunicorn", "main:app", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
