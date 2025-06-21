# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    cmake \
    make \
    git \
    wget \
    curl \
    unzip \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/models /app/temp /app/output

# Copy application files
COPY main.py .
COPY enhanced_counterfeit_detector.py .

# Copy the model files (if they exist locally)
# Note: In production, these should be downloaded from a model registry or S3
# COPY ../my_model/ ./models/

# Create a startup script
RUN echo '#!/bin/bash\n\
# Download models if they do not exist\n\
if [ ! -f "./models/my_model.pt" ]; then\n\
    echo "Model files not found. Please ensure YOLO model is available."\n\
    # mkdir -p ./models\n\
    # You can add model download logic here\n\
fi\n\
\n\
# Start the FastAPI application\n\
exec uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1 --timeout-keep-alive 60\n\
' > /app/start.sh && chmod +x /app/start.sh

# Expose port (will be overridden by AWS)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Use startup script
CMD ["/app/start.sh"] 