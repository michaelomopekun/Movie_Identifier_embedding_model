FROM python:3.10-slim AS builder

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    curl \
    xz-utils \
    && rm -rf /var/lib/apt/lists/*

# Install static ffmpeg + ffprobe
RUN curl -L https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz | tar xJ \
 && mv ffmpeg-*-static/ffmpeg /usr/local/bin/ \
 && mv ffmpeg-*-static/ffprobe /usr/local/bin/ \
 && rm -rf ffmpeg-*-static

# Install dependencies into a temp dir
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --prefix=/install torch==2.2.2 --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
 && pip install --prefix=/install -r requirements.txt --no-cache-dir


# Runtime
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgl1 && rm -rf /var/lib/apt/lists/*

# Copy dependencies and binaries
COPY --from=builder /install /usr/local
COPY --from=builder /usr/local/bin/ffmpeg /usr/local/bin/ffmpeg
COPY --from=builder /usr/local/bin/ffprobe /usr/local/bin/ffprobe

# Copy app source code
COPY . .

ENV PORT=8000
EXPOSE 8000

CMD ["gunicorn", "main:app", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]

