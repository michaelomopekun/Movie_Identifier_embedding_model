# Use slim Python image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy app files
COPY . .

# Install all Python deps
RUN pip install --upgrade pip && pip install -r requirements.txt

# Install CPU-only PyTorch
RUN pip install torch==2.2.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Clean pip cache
RUN rm -rf /root/.cache/pip

# Set port and expose it
ENV PORT=8000
EXPOSE 8000

# Start app with Gunicorn
CMD ["gunicorn", "main:app", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
